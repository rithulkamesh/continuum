#include <continuum/backend/azure_openai.hpp>

#include <curl/curl.h>
#include <continuum/utils/logging.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace continuum::backend {
namespace {

struct PrefixStateData {
  std::string prefix_text;
};

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

std::string ExtractContentField(const std::string& body) {
  const std::string key = "\"content\":";
  auto pos = body.find(key);
  if (pos == std::string::npos) return "";
  pos = body.find('"', pos + key.size());
  if (pos == std::string::npos) return "";
  ++pos;
  std::string out;
  while (pos < body.size()) {
    char c = body[pos++];
    if (c == '"' && (pos < 2 || body[pos - 2] != '\\')) break;
    out.push_back(c);
  }
  return out;
}

std::vector<int> ToTokens(const std::string& text) {
  std::vector<int> out;
  out.reserve(text.size());
  for (unsigned char c : text) out.push_back(static_cast<int>(c));
  return out;
}

std::string DeriveReusablePrefix(const std::string& prompt) {
  const std::string marker = " Question:";
  const auto pos = prompt.rfind(marker);
  if (pos == std::string::npos) {
    return prompt;
  }
  return prompt.substr(0, pos + marker.size());
}

}  // namespace

BackendCapabilities AzureOpenAIBackend::capabilities() const {
  return BackendCapabilities{false, true, true};
}

BackendRunResult AzureOpenAIBackend::run_with_cache(
    const ir::Node& node,
    const std::vector<continuum::Value>& inputs,
    const std::optional<BackendState>& prefix_state,
    std::int32_t remaining_tokens) {
  const auto* payload = std::get_if<ir::TokenOpPayload>(&node.payload);
  if (payload == nullptr) {
    throw std::runtime_error("azure backend expects TokenOpPayload");
  }
  const std::string endpoint = GetEnvOr("AZURE_OPENAI_ENDPOINT");
  const std::string api_key = GetEnvOr("AZURE_OPENAI_API_KEY");
  std::string deployment = GetEnvOr("AZURE_OPENAI_DEPLOYMENT");
  if (deployment.empty()) {
    deployment = GetEnvOr("AZURE_OPENAI_DEPLOYMENT_NAME");
  }
  if (deployment.empty()) {
    deployment = payload->model_id.empty() ? "gpt-5-mini" : payload->model_id;
  }
  const std::string api_version = GetEnvOr("AZURE_OPENAI_API_VERSION", "2024-08-01-preview");
  if (endpoint.empty() || api_key.empty()) {
    throw std::runtime_error("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set");
  }

  const std::string full_prompt = ExtractPromptText(inputs);
  std::string prefix_text;
  if (prefix_state.has_value() && prefix_state->handle != nullptr) {
    auto* ps = reinterpret_cast<PrefixStateData*>(prefix_state->handle);
    prefix_text = ps->prefix_text;
  }
  std::string suffix_text = full_prompt;
  if (!prefix_text.empty() && full_prompt.rfind(prefix_text, 0) == 0) {
    suffix_text = full_prompt.substr(prefix_text.size());
  }
  if (remaining_tokens <= 0) {
    suffix_text.clear();
  }

  const bool cache_hit = !prefix_text.empty() && full_prompt.rfind(prefix_text, 0) == 0;
  const std::string system_context = GetEnvOr("AZURE_OPENAI_SYSTEM_CONTEXT", "");
  const std::string request_prompt = cache_hit ? suffix_text : full_prompt;

  const std::string url = endpoint + "/openai/deployments/" + deployment +
                          "/chat/completions?api-version=" + api_version;
  std::ostringstream body_stream;
  body_stream << "{\"messages\":[";
  bool first_msg = true;
  if (!system_context.empty()) {
    body_stream << "{\"role\":\"system\",\"content\":\"" << JsonEscape(system_context) << "\"}";
    first_msg = false;
  }
  if (!first_msg) {
    body_stream << ",";
  }
  body_stream << "{\"role\":\"user\",\"content\":\"" << JsonEscape(request_prompt) << "\"}";
  body_stream << "]";
  if (std::abs(payload->temperature - 1.0f) < 1e-6f) {
    body_stream << ",\"temperature\":1";
  }
  body_stream << ",\"max_completion_tokens\":" << payload->max_tokens << "}";
  const std::string body = body_stream.str();

  auto start = std::chrono::steady_clock::now();
  std::string response;
  CURL* curl = curl_easy_init();
  if (curl == nullptr) throw std::runtime_error("curl init failed");
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, ("api-key: " + api_key).c_str());
  headers = curl_slist_append(headers, "Content-Type: application/json");
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
  CURLcode rc = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  auto end = std::chrono::steady_clock::now();
  const auto latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  if (rc != CURLE_OK || http_code >= 400) {
    throw std::runtime_error("azure openai request failed: code=" + std::to_string(http_code) + " body=" + response);
  }
  const std::string completion = ExtractContentField(response);
  std::string combined = full_prompt + completion;

  auto* new_state = new PrefixStateData{DeriveReusablePrefix(full_prompt)};
  BackendRunResult out;
  out.output = continuum::TokensValue{ToTokens(combined)};
  out.resulting_state.handle = reinterpret_cast<void*>(new_state);
  out.reused_prefix_len = static_cast<std::int32_t>(new_state->prefix_text.size());
  out.compute_steps = static_cast<std::int32_t>(request_prompt.size());
  out.tokens_sent = static_cast<std::int32_t>(request_prompt.size());
  out.tokens_saved = cache_hit ? static_cast<std::int32_t>(prefix_text.size()) : 0;
  out.used_cached_state = cache_hit;
  LOG_INFO(
      backend,
      "backend=azure cache_{} latency_ms={} tokens_sent={} tokens_saved={} deployment={}",
      (cache_hit ? "hit" : "miss"),
      latency_ms,
      out.tokens_sent,
      out.tokens_saved,
      deployment);
  return out;
}

}  // namespace continuum::backend
