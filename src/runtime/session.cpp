#include <continuum/runtime/session.hpp>
#include <continuum/runtime/memo_table.hpp>
#include <continuum/runtime/semantic_cache.hpp>
#include <continuum/runtime/layer_cache.hpp>
#include <continuum/runtime/memory_graph.hpp>

#include <continuum/utils/logging.hpp>

#include <cstdint>
#include <sstream>
#include <variant>

namespace continuum::runtime {

namespace {

std::vector<std::int32_t> CanonicalizeInputTokens(const std::vector<continuum::Value>& input_values) {
  if (!input_values.empty()) {
    if (const auto* tv = std::get_if<continuum::TokensValue>(&input_values.front())) {
      std::vector<std::int32_t> tokens;
      tokens.reserve(tv->ids.size());
      for (int id : tv->ids) {
        tokens.push_back(static_cast<std::int32_t>(id));
      }
      return tokens;
    }
  }
  std::string joined;
  for (std::size_t i = 0; i < input_values.size(); ++i) {
    if (i != 0) joined += " | ";
    const auto& v = input_values[i];
    if (const auto* s = std::get_if<std::string>(&v)) {
      joined += *s;
    } else if (const auto* sch = std::get_if<continuum::SchemaValue>(&v)) {
      joined += sch->json;
    } else if (const auto* d = std::get_if<double>(&v)) {
      joined += std::to_string(*d);
    } else if (const auto* n = std::get_if<std::int64_t>(&v)) {
      joined += std::to_string(*n);
    }
  }
  std::string text = joined;
  std::string out;
  out.reserve(text.size());
  bool prev_space = false;
  for (char c : text) {
    const bool is_space = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
    if (is_space) {
      if (!prev_space) { out.push_back(' '); }
      prev_space = true;
    } else {
      out.push_back(c);
      prev_space = false;
    }
  }
  while (!out.empty() && out.front() == ' ') out.erase(out.begin());
  while (!out.empty() && out.back() == ' ') out.pop_back();
  std::vector<std::int32_t> tokens;
  tokens.reserve(out.size());
  for (unsigned char c : out) {
    tokens.push_back(static_cast<std::int32_t>(c));
  }
  return tokens;
}

}  // namespace

Session::Session(const std::string& id, backend::BackendRegistry& backends,
                 std::size_t max_cache_entries)
    : session_id_(id),
      backends_(backends),
      owned_cache_(std::make_unique<KVCacheIndex>(max_cache_entries)),
      cache_(*owned_cache_),
      metrics_{} {
  metrics_.session_id = id;
}

Session::Session(const std::string& id, backend::BackendRegistry& backends,
                 KVCacheIndex& external_cache)
    : session_id_(id),
      backends_(backends),
      owned_cache_(nullptr),
      cache_(external_cache),
      metrics_{} {
  metrics_.session_id = id;
}

std::vector<continuum::Value> Session::run(
    const ir::Graph& g, const std::unordered_map<ir::NodeId, continuum::Value>& inputs) {
  std::vector<continuum::Value> results;

  for (const auto id : g.topo_order()) {
    const auto& node = g.get(id);
    if (node.kind == ir::NodeKind::TokenOp) {
      const auto* payload = std::get_if<ir::TokenOpPayload>(&node.payload);
      if (payload) {
        std::vector<continuum::Value> in_vals;
        for (auto inp_id : node.inputs) {
          auto in_it = inputs.find(inp_id);
          if (in_it != inputs.end()) {
            in_vals.push_back(in_it->second);
          }
        }
        auto input_tokens = CanonicalizeInputTokens(in_vals);
        DecodeParams dp{payload->op_name, payload->temperature, payload->max_tokens};

        metrics_.total_lookups++;
        ReuseStepRecord rec;
        rec.node_name = node.debug_name;
        rec.total_tokens = static_cast<std::int32_t>(input_tokens.size());

        bool memo_hit = false;
        bool semantic_hit = false;

        if (memo_table_ != nullptr) {
          auto memo_key = memo_table_->make_key(node, in_vals);
          auto memo_result = memo_table_->lookup(memo_key);
          if (memo_result.has_value()) {
            memo_hit = true;
            rec.memo_hit = true;
            rec.tokens_saved = rec.total_tokens;
            metrics_.total_hits++;
            metrics_.total_tokens_saved += rec.total_tokens;
          }
        }

        if (!memo_hit && semantic_cache_ != nullptr && embedder_ != nullptr) {
          std::string prompt_text;
          for (const auto& v : in_vals) {
            if (const auto* s = std::get_if<std::string>(&v)) prompt_text += *s;
          }
          if (!prompt_text.empty()) {
            auto embedding = embedder_->embed(prompt_text);
            auto sem_result = semantic_cache_->lookup(embedding, payload->model_id);
            if (sem_result.above_threshold) {
              semantic_hit = true;
              rec.semantic_hit = true;
              rec.semantic_similarity = sem_result.similarity;
              metrics_.total_hits++;
              metrics_.total_tokens_saved += rec.total_tokens;
            }
          }
        }

        if (!memo_hit && !semantic_hit) {
          auto hit = cache_.longest_prefix(payload->model_id, dp, input_tokens);
          if (hit.has_value()) {
            std::int32_t hit_len = hit->second;
            if (hit_len > static_cast<std::int32_t>(input_tokens.size())) {
              hit_len = static_cast<std::int32_t>(input_tokens.size());
            }
            if (policy_.should_attempt(input_tokens, hit_len)) {
              rec.cache_hit = true;
              rec.prefix_hit_len = hit_len;
              rec.tokens_saved = hit_len;
              metrics_.total_hits++;
              metrics_.total_tokens_saved += hit_len;
              metrics_.total_tokens_processed += static_cast<std::int64_t>(input_tokens.size()) - hit_len;
            } else {
              metrics_.total_tokens_processed += input_tokens.size();
            }
          } else {
            metrics_.total_tokens_processed += input_tokens.size();
          }
        }

        metrics_.steps.push_back(rec);
      }
    }
  }

  Interpreter interp(backends_, cache_, &policy_);
  interp.set_memo_table(memo_table_);
  interp.set_semantic_cache(semantic_cache_);
  interp.set_embedding_provider(embedder_);
  interp.set_layer_cache(layer_cache_);
  interp.set_memory_graph(memory_graph_);
  results = interp.run(g, inputs);
  metrics_.run_count++;

  return results;
}

bool Session::save_cache_metadata(const std::string& path) const {
  return cache_.save_metadata(path);
}

bool Session::load_cache_metadata(const std::string& path) {
  return cache_.load_metadata(path);
}

}  // namespace continuum::runtime
