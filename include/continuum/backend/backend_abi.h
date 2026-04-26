#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CONTINUUM_BACKEND_ABI_VERSION 1u

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
} continuum_backend_vtable_t;

#ifdef __cplusplus
}
#endif
