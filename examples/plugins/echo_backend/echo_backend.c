/*
 * Echo backend: a minimal out-of-tree Continuum backend plugin.
 *
 * Built against only <continuum/backend/backend_abi.h> (plain C, no Continuum
 * libraries, no libtorch). It answers TokenOps with "echo: <prompt>", hands
 * out integer state handles for prefix reuse, and reports metrics that meet
 * the backend conformance contract (docs/design/abi.md).
 */
#include <continuum/backend/backend_abi.h>

#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define ECHO_EXPORT __declspec(dllexport)
#else
#define ECHO_EXPORT __attribute__((visibility("default")))
#endif

typedef struct echo_instance_s {
  char* output;           /* last output; valid until the next call */
  size_t output_cap;
  uint64_t next_state;    /* state handles are 1, 2, 3, ... */
} echo_instance_t;

static continuum_backend_caps_t echo_capabilities(void* instance) {
  (void)instance;
  continuum_backend_caps_t caps = {0, 1, 1}; /* token + cache, no tensor */
  return caps;
}

static const char* echo_tensor_backend_type(void* instance) {
  (void)instance;
  return "";
}

static int echo_reserve(echo_instance_t* self, size_t need) {
  if (need <= self->output_cap) return 1;
  char* grown = (char*)realloc(self->output, need);
  if (grown == NULL) return 0;
  self->output = grown;
  self->output_cap = need;
  return 1;
}

static continuum_backend_run_result_t echo_run(void* instance, continuum_backend_node_meta_t node,
                                               const continuum_backend_value_t* inputs, size_t input_count,
                                               const continuum_backend_state_t* prefix_state,
                                               int32_t remaining_tokens) {
  echo_instance_t* self = (echo_instance_t*)instance;
  continuum_backend_run_result_t r;
  memset(&r, 0, sizeof(r));
  (void)node;

  /* Prompt = string inputs joined with '\n'; token inputs count as bytes. */
  size_t prompt_len = 0;
  for (size_t i = 0; i < input_count; ++i) {
    if (i != 0) prompt_len += 1;
    if (inputs[i].kind == CONTINUUM_BACKEND_VALUE_STRING && inputs[i].string_data != NULL) {
      prompt_len += strlen(inputs[i].string_data);
    } else if (inputs[i].kind == CONTINUUM_BACKEND_VALUE_TOKENS) {
      prompt_len += inputs[i].token_count;
    }
  }
  static const char prefix[] = "echo: ";
  if (!echo_reserve(self, sizeof(prefix) + prompt_len + 1)) {
    return r; /* NONE output on allocation failure */
  }
  char* w = self->output;
  memcpy(w, prefix, sizeof(prefix) - 1);
  w += sizeof(prefix) - 1;
  for (size_t i = 0; i < input_count; ++i) {
    if (i != 0) *w++ = '\n';
    if (inputs[i].kind == CONTINUUM_BACKEND_VALUE_STRING && inputs[i].string_data != NULL) {
      size_t n = strlen(inputs[i].string_data);
      memcpy(w, inputs[i].string_data, n);
      w += n;
    } else if (inputs[i].kind == CONTINUUM_BACKEND_VALUE_TOKENS) {
      for (size_t t = 0; t < inputs[i].token_count; ++t) *w++ = (char)inputs[i].token_ids[t];
    }
  }
  *w = '\0';

  const int32_t total = (int32_t)prompt_len;
  int32_t remaining = remaining_tokens < 0 ? 0 : remaining_tokens;
  if (remaining > total) remaining = total;
  const int warm = prefix_state != NULL && prefix_state->handle != NULL;

  r.output.kind = CONTINUUM_BACKEND_VALUE_STRING;
  r.output.string_data = self->output;
  r.resulting_state.handle = (void*)(uintptr_t)(++self->next_state);
  r.reused_prefix_len = warm ? total - remaining : 0;
  r.tokens_saved = r.reused_prefix_len;
  r.tokens_sent = warm ? remaining : total;
  r.compute_steps = r.tokens_sent;
  r.used_cached_state = (uint8_t)warm;
  return r;
}

static void echo_destroy(void* instance) {
  echo_instance_t* self = (echo_instance_t*)instance;
  free(self->output);
  free(self);
}

static size_t echo_export_state(void* instance, continuum_backend_state_t state, uint8_t* out, size_t out_cap) {
  (void)instance;
  const uint64_t id = (uint64_t)(uintptr_t)state.handle;
  if (id == 0) return 0;
  if (out != NULL && out_cap >= sizeof(id)) memcpy(out, &id, sizeof(id));
  return sizeof(id);
}

static uint8_t echo_import_state(void* instance, const uint8_t* bytes, size_t len, continuum_backend_state_t* out) {
  (void)instance;
  uint64_t id = 0;
  if (len != sizeof(id)) return 0;
  memcpy(&id, bytes, sizeof(id));
  if (id == 0) return 0;
  out->handle = (void*)(uintptr_t)id;
  return 1;
}

ECHO_EXPORT int continuum_backend_plugin_init_v2(uint32_t host_abi_version, continuum_backend_vtable_t* out) {
  if (host_abi_version < 2u) return 1; /* needs the v2 destroy / state hooks */
  echo_instance_t* self = (echo_instance_t*)calloc(1, sizeof(echo_instance_t));
  if (self == NULL) return 2;
  out->abi_version = 2u;
  out->instance = self;
  out->capabilities = echo_capabilities;
  out->tensor_backend_type = echo_tensor_backend_type;
  out->run_with_cache = echo_run;
  out->destroy = echo_destroy;
  out->export_state = echo_export_state;
  out->import_state = echo_import_state;
  return 0;
}
