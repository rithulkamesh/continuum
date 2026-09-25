#pragma once

/*
 * Continuum backend C ABI.
 *
 * A backend compiled against only this header can be loaded at runtime:
 * build it as a shared library that exports
 * CONTINUUM_BACKEND_PLUGIN_INIT_SYMBOL (see continuum_backend_plugin_init_fn)
 * and point the runtime at it with BackendRegistry::load_plugin or the
 * CONTINUUM_BACKEND_PLUGINS environment variable. See docs/design/abi.md.
 *
 * Memory: every pointer the host passes in is borrowed for the duration of
 * the call. Pointers a backend returns (output strings / token arrays, the
 * tensor_backend_type name) must stay valid until the next call on the same
 * instance; the host copies them immediately.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* v1: capabilities + run_with_cache. v2 appends destroy and optional state
 * portability to the vtable; the host still accepts v1 vtables. */
#define CONTINUUM_BACKEND_ABI_VERSION 2u
#define CONTINUUM_BACKEND_ABI_MIN_VERSION 1u

/* Versioned plugin entry point; the suffix is the ABI major it implements. */
#define CONTINUUM_BACKEND_PLUGIN_INIT_SYMBOL "continuum_backend_plugin_init_v2"

typedef enum continuum_backend_value_kind_e {
  CONTINUUM_BACKEND_VALUE_NONE = 0,
  CONTINUUM_BACKEND_VALUE_STRING = 1,
  CONTINUUM_BACKEND_VALUE_TOKENS = 2
} continuum_backend_value_kind_t;

typedef struct continuum_backend_caps_s {
  uint8_t supports_tensor;
  uint8_t supports_token;
  uint8_t supports_cache;
} continuum_backend_caps_t;

typedef struct continuum_backend_state_s {
  void* handle;
} continuum_backend_state_t;

typedef struct continuum_backend_value_s {
  continuum_backend_value_kind_t kind;
  const char* string_data;
  const int32_t* token_ids;
  size_t token_count;
} continuum_backend_value_t;

typedef struct continuum_backend_run_result_s {
  continuum_backend_value_t output;
  continuum_backend_state_t resulting_state;
  int32_t reused_prefix_len;
  int32_t compute_steps;
  int32_t tokens_sent;
  int32_t tokens_saved;
  uint8_t used_cached_state;
} continuum_backend_run_result_t;

typedef struct continuum_backend_node_meta_s {
  uint8_t node_kind;
  const char* op_name;
  const char* model_id;
  const int64_t* attrs;
  size_t attr_count;
} continuum_backend_node_meta_t;

typedef struct continuum_backend_vtable_s {
  uint32_t abi_version;
  void* instance;
  continuum_backend_caps_t (*capabilities)(void* instance);
  const char* (*tensor_backend_type)(void* instance);
  continuum_backend_run_result_t (*run_with_cache)(
      void* instance,
      continuum_backend_node_meta_t node,
      const continuum_backend_value_t* inputs,
      size_t input_count,
      const continuum_backend_state_t* prefix_state,
      int32_t remaining_tokens);

  /* ---- v2 (optional; leave NULL when unsupported) ---- */

  /* Called once when the host drops the backend. Frees `instance`. */
  void (*destroy)(void* instance);
  /* Serialize `state` into `out` (capacity `out_cap`) and return the number
   * of bytes required. The host calls it with out == NULL to size the
   * buffer. Returning 0 means the state is not portable. */
  size_t (*export_state)(void* instance, continuum_backend_state_t state, uint8_t* out, size_t out_cap);
  /* Rebuild a live state from exported bytes. Returns 1 and fills *out on
   * success, 0 if the bytes cannot be imported. */
  uint8_t (*import_state)(void* instance, const uint8_t* bytes, size_t len, continuum_backend_state_t* out);
} continuum_backend_vtable_t;

/*
 * Plugin entry point, exported under CONTINUUM_BACKEND_PLUGIN_INIT_SYMBOL.
 * The host zero-fills *out, passes its own CONTINUUM_BACKEND_ABI_VERSION, and
 * expects the plugin to fill *out (including abi_version and instance) and
 * return 0. A non-zero return aborts loading.
 */
typedef int (*continuum_backend_plugin_init_fn)(uint32_t host_abi_version,
                                                continuum_backend_vtable_t* out);

#ifdef __cplusplus
}
#endif
