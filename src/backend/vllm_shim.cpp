#include <continuum/backend/vllm_shim.hpp>

#include <curl/curl.h>
#include <continuum/utils/logging.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace continuum::backend {
namespace {

std::string GetEnvOr(const char* key, const std::string& fallback = "") {
  const char* v = std::getenv(key);
  return v == nullptr ? fallback : std::string(v);
}

std::string CanonicalizeText(const std::string& raw) {
  std::string out;
  out.reserve(raw.size());
  bool prev_space = false;
  for (char c : raw) {
    const bool is_space = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
    if (is_space) {
      if (!prev_space) out.push_back(' ');
      prev_space = true;
    } else {
      out.push_back(c);
      prev_space = false;
    }
  }
  while (!out.empty() && out.front() == ' ') out.erase(out.begin());
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

std::string ExtractPromptText(const std::vector<continuum::Value>& inputs) {
  std::ostringstream ss;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (i != 0) ss << "\n";
    const auto& v = inputs[i];
    if (const auto* s = std::get_if<std::string>(&v)) {
      ss << *s;
    } else if (const auto* t = std::get_if<continuum::TokensValue>(&v)) {
      for (int id : t->ids) ss << static_cast<char>(id);
    } else if (const auto* s = std::get_if<continuum::SchemaValue>(&v)) {
      ss << s->json;
    } else if (const auto* d = std::get_if<double>(&v)) {
      ss << *d;
    } else if (const auto* n = std::get_if<int64_t>(&v)) {
      ss << *n;
    }
  }
  return CanonicalizeText(ss.str());
}

std::size_t WriteCb(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
  auto* s = reinterpret_cast<std::string*>(userdata);
  s->append(ptr, size * nmemb);
  return size * nmemb;
}

std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 16);
  for (char c : s) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

void AppendUtf8(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

bool ParseHex4(const std::string& s, std::size_t pos, std::uint32_t& out) {
  if (pos + 4 > s.size()) return false;
  out = 0;
  for (std::size_t i = pos; i < pos + 4; ++i) {
    const char c = s[i];
    out <<= 4;
    if (c >= '0' && c <= '9') out |= static_cast<std::uint32_t>(c - '0');
    else if (c >= 'a' && c <= 'f') out |= static_cast<std::uint32_t>(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') out |= static_cast<std::uint32_t>(c - 'A' + 10);
    else return false;
  }
  return true;
}

}  // namespace

std::string VllmShimBackend::ExtractJsonString(const std::string& body, const std::string& key) {
  const std::string quoted = "\"" + key + "\"";
  auto pos = body.find(quoted);
  if (pos == std::string::npos) return "";
  pos = body.find(':', pos + quoted.size());
  if (pos == std::string::npos) return "";
  pos = body.find('"', pos + 1);
  if (pos == std::string::npos) return "";
  ++pos;
  std::string out;
  while (pos < body.size()) {
    const char c = body[pos++];
    if (c == '"') break;
    if (c != '\\' || pos >= body.size()) {
      out.push_back(c);
      continue;
    }
    const char esc = body[pos++];
    switch (esc) {
      case 'n': out.push_back('\n'); break;
      case 't': out.push_back('\t'); break;
      case 'r': out.push_back('\r'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'u': {
        std::uint32_t cp = 0;
        if (!ParseHex4(body, pos, cp)) return out;
        pos += 4;
        // Surrogate pair: a high surrogate must be followed by \uDC00-\uDFFF.
        std::uint32_t low = 0;
        if (cp >= 0xD800 && cp <= 0xDBFF && pos + 6 <= body.size() && body[pos] == '\\' &&
            body[pos + 1] == 'u' && ParseHex4(body, pos + 2, low) && low >= 0xDC00 && low <= 0xDFFF) {
          cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
          pos += 6;
        }
        AppendUtf8(out, cp);
        break;
      }
      default: out.push_back(esc); break;  // \" \\ \/
    }
  }
  return out;
}

std::int32_t VllmShimBackend::ExtractJsonInt(const std::string& body, const std::string& key) {
  const std::string quoted = "\"" + key + "\"";
  auto pos = body.find(quoted);
  if (pos == std::string::npos) return 0;
  pos = body.find(':', pos + quoted.size());
  if (pos == std::string::npos) return 0;
  ++pos;
  while (pos < body.size() && body[pos] == ' ') ++pos;
  std::int64_t v = 0;
  bool any = false;
  while (pos < body.size() && body[pos] >= '0' && body[pos] <= '9') {
    v = v * 10 + (body[pos++] - '0');
    any = true;
    if (v > INT32_MAX) return INT32_MAX;
  }
  return any ? static_cast<std::int32_t>(v) : 0;
}

namespace {

std::string DeriveReusablePrefix(const std::string& prompt) {
  const std::string marker = " Question:";
  const auto pos = prompt.rfind(marker);
  if (pos == std::string::npos) {
    return prompt;
  }
  return prompt.substr(0, pos + marker.size());
}

continuum::TokensValue GenerateDeterministicOutput(const std::string& prompt, std::int32_t max_tokens) {
  continuum::TokensValue out;
  out.ids.reserve(static_cast<std::size_t>(std::max<std::int32_t>(0, max_tokens)));
  for (std::int32_t i = 0; i < max_tokens; ++i) {
    const int base = prompt.empty() ? 0 : static_cast<unsigned char>(prompt[static_cast<std::size_t>(i) % prompt.size()]);
    out.ids.push_back((base + 43 + i) % 10037);
  }
  return out;
}

}  // namespace

namespace {

// Portable prefix-state wire format: magic "VPS1", then length-prefixed
// prefix text and model name. It records which prefix the server was last
// given, so a resumed process can credit (and optionally re-warm) it.
constexpr std::uint32_t kStateMagic = 0x31535056U;  // "VPS1"

void PutU32(std::vector<std::uint8_t>& out, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
}

bool GetU32(const std::vector<std::uint8_t>& in, std::size_t& pos, std::uint32_t& v) {
  if (pos + 4 > in.size()) return false;
  v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(in[pos + static_cast<std::size_t>(i)]) << (8 * i);
  pos += 4;
  return true;
}

void PutStr(std::vector<std::uint8_t>& out, const std::string& s) {
  PutU32(out, static_cast<std::uint32_t>(s.size()));
  out.insert(out.end(), s.begin(), s.end());
}

bool GetStr(const std::vector<std::uint8_t>& in, std::size_t& pos, std::string& s) {
  std::uint32_t n = 0;
  if (!GetU32(in, pos, n) || pos + n > in.size()) return false;
  s.assign(reinterpret_cast<const char*>(in.data() + pos), n);
  pos += n;
  return true;
}

std::string ModelName(const std::string& model_id) {
  return model_id.rfind("vllm/", 0) == 0 ? model_id.substr(5) : GetEnvOr("VLLM_MODEL", model_id);
}

// POST {base}/v1/completions and return the body; throws on transport or HTTP errors.
std::string PostCompletion(const std::string& base, const std::string& model, const std::string& prompt,
                           std::int32_t max_tokens, float temperature) {
  std::ostringstream body_stream;
  body_stream << "{\"model\":\"" << JsonEscape(model) << "\",\"prompt\":\"" << JsonEscape(prompt)
              << "\",\"max_tokens\":" << max_tokens << ",\"temperature\":" << temperature << "}";
  const std::string body = body_stream.str();
  const std::string url = base + "/v1/completions";
  std::string response;
  CURL* curl = curl_easy_init();
  if (curl == nullptr) throw std::runtime_error("curl init failed");
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
  const CURLcode rc = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  if (rc != CURLE_OK || http_code >= 400) {
    throw std::runtime_error("vllm request failed: code=" + std::to_string(http_code) + " body=" + response);
  }
  return response;
}

}  // namespace

BackendCapabilities VllmShimBackend::capabilities() const {
  return BackendCapabilities{false, true, true};
}

BackendState VllmShimBackend::remember(std::string prefix_text, std::string model) {
  std::lock_guard<std::mutex> lock(mu_);
  const std::uint64_t id = next_state_id_++;
  states_[id] = PrefixState{std::move(prefix_text), std::move(model)};
  // Bounded: dropping the oldest handles only turns a later lookup into a
  // cold (full-prompt) request, never a wrong one.
  while (states_.size() > kMaxStates) {
    states_.erase(states_.begin());
  }
  return BackendState{reinterpret_cast<void*>(static_cast<std::uintptr_t>(id))};
}

std::optional<VllmShimBackend::PrefixState> VllmShimBackend::lookup(const BackendState& state) const {
  const auto id = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(state.handle));
  std::lock_guard<std::mutex> lock(mu_);
  auto it = states_.find(id);
  if (it == states_.end()) return std::nullopt;
  return it->second;
}

std::size_t VllmShimBackend::state_count() const {
  std::lock_guard<std::mutex> lock(mu_);
  return states_.size();
}

std::vector<std::uint8_t> VllmShimBackend::export_state(const BackendState& state) const {
  const auto ps = lookup(state);
  if (!ps.has_value()) return {};
  std::vector<std::uint8_t> out;
  PutU32(out, kStateMagic);
  PutStr(out, ps->prefix_text);
  PutStr(out, ps->model);
  return out;
}

std::optional<BackendState> VllmShimBackend::import_state(const std::vector<std::uint8_t>& bytes) {
  std::size_t pos = 0;
  std::uint32_t magic = 0;
  PrefixState ps;
  if (!GetU32(bytes, pos, magic) || magic != kStateMagic || !GetStr(bytes, pos, ps.prefix_text) ||
      !GetStr(bytes, pos, ps.model) || pos != bytes.size()) {
    return std::nullopt;
  }
  // The KV blocks live in the server; after a server restart they are gone.
  // Opt-in re-warm: one 1-token request over the prefix repopulates the
  // server's prefix cache before the resumed run needs it.
  const std::string base = GetEnvOr("VLLM_BASE_URL", "");
  if (!base.empty() && !ps.prefix_text.empty() && GetEnvOr("VLLM_REWARM_ON_IMPORT", "") == "1") {
    try {
      PostCompletion(base, ps.model, ps.prefix_text, 1, 0.0f);
      rewarm_count_.fetch_add(1);
    } catch (const std::exception& e) {
      LOG_INFO(backend, "backend=vllm rewarm_failed error={}", e.what());
    }
  }
  return remember(std::move(ps.prefix_text), std::move(ps.model));
}

BackendRunResult VllmShimBackend::run_with_cache(
    const ir::Node& node,
    const std::vector<continuum::Value>& inputs,
    const std::optional<BackendState>& prefix_state,
    std::int32_t remaining_tokens) {
  const auto* payload = std::get_if<ir::TokenOpPayload>(&node.payload);
  if (payload == nullptr) {
    throw std::runtime_error("vllm backend expects TokenOpPayload");
  }
  const std::string full_prompt = ExtractPromptText(inputs);
  std::string prefix_text;
  if (prefix_state.has_value() && prefix_state->handle != nullptr) {
    if (auto ps = lookup(*prefix_state)) prefix_text = std::move(ps->prefix_text);
  }
  const bool cache_hit = !prefix_text.empty() && full_prompt.rfind(prefix_text, 0) == 0;
  const std::string model = ModelName(payload->model_id);
  const std::string base = GetEnvOr("VLLM_BASE_URL", "");
  if (base.empty()) {
    // Offline: deterministic output, suffix-only accounting.
    std::string request_prompt = cache_hit ? full_prompt.substr(prefix_text.size()) : full_prompt;
    if (remaining_tokens <= 0) request_prompt.clear();
    BackendRunResult out;
    out.output = GenerateDeterministicOutput(full_prompt, payload->max_tokens);
    out.resulting_state = remember(DeriveReusablePrefix(full_prompt), model);
    out.reused_prefix_len = cache_hit ? static_cast<std::int32_t>(prefix_text.size()) : 0;
    out.compute_steps = static_cast<std::int32_t>(request_prompt.size());
    out.tokens_sent = static_cast<std::int32_t>(request_prompt.size());
    out.tokens_saved = cache_hit ? static_cast<std::int32_t>(prefix_text.size()) : 0;
    out.used_cached_state = cache_hit;
    return out;
  }

  // A real server needs the whole prompt: it has no view of Continuum's
  // prefix state, and a suffix-only request would drop the shared context.
  // Servers with automatic prefix caching (vLLM APC, Ollama's KV reuse) skip
  // recomputing the shared prefix themselves and report it as cached_tokens.
  const auto start = std::chrono::steady_clock::now();
  const std::string response = PostCompletion(base, model, full_prompt, payload->max_tokens, payload->temperature);
  const auto latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

  const std::string completion = ExtractJsonString(response, "text");
  const std::int32_t server_cached = ExtractJsonInt(response, "cached_tokens");

  BackendRunResult out;
  out.output = completion;
  out.resulting_state = remember(DeriveReusablePrefix(full_prompt), model);
  out.reused_prefix_len = cache_hit ? static_cast<std::int32_t>(prefix_text.size()) : 0;
  out.compute_steps = static_cast<std::int32_t>(full_prompt.size());
  out.tokens_sent = static_cast<std::int32_t>(full_prompt.size());
  // Prefer the server's own prefix-cache report; fall back to Continuum's
  // prefix hit (bytes of shared prompt the server could have reused).
  out.tokens_saved = server_cached > 0 ? server_cached
                                       : (cache_hit ? static_cast<std::int32_t>(prefix_text.size()) : 0);
  out.used_cached_state = cache_hit || server_cached > 0;
  LOG_INFO(
      backend,
      "backend=vllm cache_{} latency_ms={} tokens_sent={} tokens_saved={} server_cached_tokens={} model={}",
      (cache_hit ? "hit" : "miss"),
      latency_ms,
      out.tokens_sent,
      out.tokens_saved,
      server_cached,
      model);
  return out;
}

}  // namespace continuum::backend
