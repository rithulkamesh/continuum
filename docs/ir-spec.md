# CIR Spec v0.1

Canonical IR contract for Continuum v0.1.

## Canonical Schema

The canonical machine-readable schema lives at `schema/cir.fbs`.

## Binary Envelope (runtime wire format)

`Graph::serialize()` currently emits a compact binary envelope.

```
u32 magic = 0x31495243   // "CIR1"
u16 version = 1
u64 node_count
repeat node_count times:
  u64 node_id
  u8  node_kind
  string debug_name            // u64 len + bytes
  u64 input_count
  u64[input_count] inputs
  Type out_type                // tagged union
  u8  effect_bits
  Payload payload              // tagged union
```

Tagged unions are encoded as:

- `Type`: `u8 idx` then variant payload
  - `0` TensorType: `u64 ndim`, `i64[ndim] shape`, `u8 dtype`, `u8 device`
  - `1` TokensType: `string vocab_id`, `i32 max_len`, `string model_family`
  - `2` SchemaType: `string canonical_json`, `u64 schema_hash`
  - `3` EffectType: `u8 bits`
- `Payload`: `u8 idx` then variant payload
  - `0` TensorOpPayload: `string op_name`, `u64 n_attrs`, `i64[n_attrs] attrs`
  - `1` TokenOpPayload: `string op_name`, `string model_id`, `f32 temperature`, `i32 max_tokens`
  - `2` PromptOpPayload: `string template_id`, `u64 n_slots`, `u64[n_slots] slot_inputs`
  - `3` ToolOpPayload: `string tool_name`, `string input_schema_json`, `string output_schema_json`
  - `4` ControlOpPayload: `u8 kind`, `u64 n_branches`, `u64[n_branches] branch_entries`

## Enumerations

- Node kinds: TensorOp, TokenOp, PromptOp, ToolOp, ControlOp
- Type lattice: TensorType, TokensType, SchemaType, EffectType
- Effects: Pure, Idem, Net, Mut, Stoch bitset

## Verification

Conformance is enforced by C++ tests:

- serialize -> deserialize -> semantic equality checks
- binary layout parser checks for envelope/tag correctness
