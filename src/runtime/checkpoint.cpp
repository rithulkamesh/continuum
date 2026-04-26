#include <continuum/runtime/checkpoint.hpp>

#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace continuum::runtime {
namespace {

constexpr std::uint32_t kCheckpointMagic = 0x31545043U;  // "CPT1"
constexpr std::uint16_t kCheckpointVersion = 1;
constexpr std::uint8_t kValueTagTensor = 0;
constexpr std::uint8_t kValueTagTokens = 1;
constexpr std::uint8_t kValueTagSchema = 2;
constexpr std::uint8_t kValueTagString = 3;
constexpr std::uint8_t kValueTagDouble = 4;
constexpr std::uint8_t kValueTagInt64 = 5;
constexpr std::uint8_t kValueTagMlxTensor = 6;

template <typename T>
void WritePrimitive(std::vector<std::uint8_t>& out, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>, "primitive write requires trivially copyable type");
  const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
  out.insert(out.end(), begin, begin + sizeof(T));
}

template <typename T>
T ReadPrimitive(const std::uint8_t*& cur, const std::uint8_t* end) {
  if (static_cast<std::size_t>(end - cur) < sizeof(T)) {
    throw std::runtime_error("checkpoint deserialize: unexpected eof");
  }
  T out{};
  std::memcpy(&out, cur, sizeof(T));
  cur += sizeof(T);
  return out;
}

}  // namespace

std::vector<std::uint8_t> checkpoint_graph(const ir::Graph& graph) {
  return graph.serialize();
}

ir::Graph restore_graph(const std::vector<std::uint8_t>& bytes) {
  if (bytes.empty()) {
    return ir::Graph();
  }
  return ir::Graph::deserialize(bytes.data(), bytes.size());
}

std::vector<std::uint8_t> serialize_value(const continuum::Value& value) {
  std::vector<std::uint8_t> out;
  const auto tag = std::visit(
      [](const auto& v) -> std::uint8_t {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, continuum::TensorValue>) {
          return kValueTagTensor;
        } else if constexpr (std::is_same_v<T, continuum::TokensValue>) {
          return kValueTagTokens;
        } else if constexpr (std::is_same_v<T, continuum::SchemaValue>) {
          return kValueTagSchema;
        } else if constexpr (std::is_same_v<T, std::string>) {
          return kValueTagString;
        } else if constexpr (std::is_same_v<T, double>) {
          return kValueTagDouble;
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return kValueTagInt64;
        } else if constexpr (std::is_same_v<T, continuum::MlxTensorValue>) {
          return kValueTagMlxTensor;
        } else {
          static_assert(!sizeof(T*), "unsupported value type");
        }
      },
      value);
  WritePrimitive(out, tag);
  std::visit(
      [&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, continuum::TensorValue>) {
          const auto backend_len = static_cast<std::uint64_t>(v.backend_type.size());
          WritePrimitive(out, backend_len);
          out.insert(out.end(), v.backend_type.begin(), v.backend_type.end());
          auto t = v.tensor.contiguous().cpu();
          const auto dtype = static_cast<std::int32_t>(t.scalar_type());
          const auto device = static_cast<std::int32_t>(t.device().type());
          WritePrimitive(out, dtype);
          WritePrimitive(out, device);
          const auto dim = static_cast<std::uint64_t>(t.dim());
          WritePrimitive(out, dim);
          for (std::int64_t i = 0; i < t.dim(); ++i) {
            const auto d = static_cast<std::int64_t>(t.size(i));
            WritePrimitive(out, d);
          }
          const auto nbytes = static_cast<std::uint64_t>(t.numel() * t.element_size());
          WritePrimitive(out, nbytes);
          const auto* ptr = reinterpret_cast<const std::uint8_t*>(t.data_ptr());
          out.insert(out.end(), ptr, ptr + nbytes);
        } else if constexpr (std::is_same_v<T, continuum::TokensValue>) {
          const auto n = static_cast<std::uint64_t>(v.ids.size());
          WritePrimitive(out, n);
          for (int id : v.ids) {
            WritePrimitive(out, static_cast<std::int32_t>(id));
          }
        } else if constexpr (std::is_same_v<T, continuum::SchemaValue>) {
          const auto n = static_cast<std::uint64_t>(v.json.size());
          WritePrimitive(out, n);
          out.insert(out.end(), v.json.begin(), v.json.end());
        } else if constexpr (std::is_same_v<T, std::string>) {
          const auto n = static_cast<std::uint64_t>(v.size());
          WritePrimitive(out, n);
          out.insert(out.end(), v.begin(), v.end());
        } else if constexpr (std::is_same_v<T, double>) {
          WritePrimitive(out, v);
        } else if constexpr (std::is_same_v<T, int64_t>) {
          WritePrimitive(out, v);
        } else if constexpr (std::is_same_v<T, continuum::MlxTensorValue>) {
          const auto backend_len = static_cast<std::uint64_t>(v.backend_type.size());
          WritePrimitive(out, backend_len);
          out.insert(out.end(), v.backend_type.begin(), v.backend_type.end());
          const auto dim = static_cast<std::uint64_t>(v.shape.size());
          WritePrimitive(out, dim);
          for (std::int64_t d : v.shape) {
            WritePrimitive(out, d);
          }
          const auto n = static_cast<std::uint64_t>(v.data.size());
          WritePrimitive(out, n);
          for (float x : v.data) {
            WritePrimitive(out, x);
          }
        }
      },
      value);
  return out;
}

continuum::Value deserialize_value(const std::uint8_t* data, std::size_t len) {
  if (data == nullptr || len == 0) {
    throw std::runtime_error("checkpoint deserialize value: empty buffer");
  }
  const std::uint8_t* cur = data;
  const std::uint8_t* end = data + len;
  const auto tag = ReadPrimitive<std::uint8_t>(cur, end);
  switch (tag) {
    case kValueTagTensor: {
      const auto backend_len = ReadPrimitive<std::uint64_t>(cur, end);
      if (static_cast<std::size_t>(end - cur) < backend_len) {
        throw std::runtime_error("checkpoint deserialize value: tensor backend payload truncated");
      }
      std::string backend_type(reinterpret_cast<const char*>(cur), static_cast<std::size_t>(backend_len));
      cur += backend_len;
      const auto dtype = static_cast<torch::ScalarType>(ReadPrimitive<std::int32_t>(cur, end));
      (void)ReadPrimitive<std::int32_t>(cur, end);  // backend/device state is not restored in v0.1
      const auto dim = ReadPrimitive<std::uint64_t>(cur, end);
      std::vector<std::int64_t> shape;
      shape.reserve(static_cast<std::size_t>(dim));
      for (std::uint64_t i = 0; i < dim; ++i) {
        shape.push_back(ReadPrimitive<std::int64_t>(cur, end));
      }
      const auto nbytes = ReadPrimitive<std::uint64_t>(cur, end);
      if (static_cast<std::size_t>(end - cur) < nbytes) {
        throw std::runtime_error("checkpoint deserialize value: tensor payload truncated");
      }
      auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
      auto t = torch::empty(shape, options);
      std::memcpy(t.data_ptr(), cur, static_cast<std::size_t>(nbytes));
      cur += nbytes;
      return continuum::TensorValue{t, std::move(backend_type)};
    }
    case kValueTagTokens: {
      const auto n = ReadPrimitive<std::uint64_t>(cur, end);
      continuum::TokensValue tv;
      tv.ids.reserve(static_cast<std::size_t>(n));
      for (std::uint64_t i = 0; i < n; ++i) {
        tv.ids.push_back(static_cast<int>(ReadPrimitive<std::int32_t>(cur, end)));
      }
      return tv;
    }
    case kValueTagSchema: {
      const auto n = ReadPrimitive<std::uint64_t>(cur, end);
      if (static_cast<std::size_t>(end - cur) < n) {
        throw std::runtime_error("checkpoint deserialize value: schema payload truncated");
      }
      continuum::SchemaValue sv;
      sv.json.assign(reinterpret_cast<const char*>(cur), static_cast<std::size_t>(n));
      cur += n;
      return sv;
    }
    case kValueTagString: {
      const auto n = ReadPrimitive<std::uint64_t>(cur, end);
      if (static_cast<std::size_t>(end - cur) < n) {
        throw std::runtime_error("checkpoint deserialize value: string payload truncated");
      }
      std::string s(reinterpret_cast<const char*>(cur), static_cast<std::size_t>(n));
      cur += n;
      return s;
    }
    case kValueTagDouble:
      return ReadPrimitive<double>(cur, end);
    case kValueTagInt64:
      return ReadPrimitive<int64_t>(cur, end);
    case kValueTagMlxTensor: {
      const auto backend_len = ReadPrimitive<std::uint64_t>(cur, end);
      if (static_cast<std::size_t>(end - cur) < backend_len) {
        throw std::runtime_error("checkpoint deserialize value: mlx backend payload truncated");
      }
      std::string backend_type(reinterpret_cast<const char*>(cur), static_cast<std::size_t>(backend_len));
      cur += backend_len;
      const auto dim = ReadPrimitive<std::uint64_t>(cur, end);
      continuum::MlxTensorValue out;
      out.shape.reserve(static_cast<std::size_t>(dim));
      for (std::uint64_t i = 0; i < dim; ++i) {
        out.shape.push_back(ReadPrimitive<std::int64_t>(cur, end));
      }
      const auto n = ReadPrimitive<std::uint64_t>(cur, end);
      out.data.reserve(static_cast<std::size_t>(n));
      for (std::uint64_t i = 0; i < n; ++i) {
        out.data.push_back(ReadPrimitive<float>(cur, end));
      }
      out.backend_type = std::move(backend_type);
      return out;
    }
    default:
      throw std::runtime_error("checkpoint deserialize value: unknown tag");
  }
}

std::vector<std::uint8_t> serialize_checkpoint(const Checkpoint& checkpoint) {
  std::vector<std::uint8_t> out;
  WritePrimitive(out, kCheckpointMagic);
  WritePrimitive(out, kCheckpointVersion);
  WritePrimitive(out, static_cast<std::uint64_t>(checkpoint.serialized_graph.size()));
  out.insert(out.end(), checkpoint.serialized_graph.begin(), checkpoint.serialized_graph.end());
  WritePrimitive(out, checkpoint.current_node_index);
  WritePrimitive(out, static_cast<std::uint64_t>(checkpoint.value_map.size()));
  for (const auto& [id, value] : checkpoint.value_map) {
    WritePrimitive(out, id);
    auto vbytes = serialize_value(value);
    WritePrimitive(out, static_cast<std::uint64_t>(vbytes.size()));
    out.insert(out.end(), vbytes.begin(), vbytes.end());
  }
  return out;
}

Checkpoint deserialize_checkpoint(const std::vector<std::uint8_t>& bytes) {
  if (bytes.empty()) {
    throw std::runtime_error("checkpoint deserialize: empty buffer");
  }
  const std::uint8_t* cur = bytes.data();
  const std::uint8_t* end = cur + bytes.size();
  const auto magic = ReadPrimitive<std::uint32_t>(cur, end);
  const auto version = ReadPrimitive<std::uint16_t>(cur, end);
  if (magic != kCheckpointMagic || version != kCheckpointVersion) {
    throw std::runtime_error("checkpoint deserialize: unsupported format");
  }
  Checkpoint out;
  const auto graph_len = ReadPrimitive<std::uint64_t>(cur, end);
  if (static_cast<std::size_t>(end - cur) < graph_len) {
    throw std::runtime_error("checkpoint deserialize: graph payload truncated");
  }
  out.serialized_graph.insert(out.serialized_graph.end(), cur, cur + graph_len);
  cur += graph_len;
  out.current_node_index = ReadPrimitive<std::uint64_t>(cur, end);
  const auto value_count = ReadPrimitive<std::uint64_t>(cur, end);
  for (std::uint64_t i = 0; i < value_count; ++i) {
    const auto id = ReadPrimitive<ir::NodeId>(cur, end);
    const auto value_len = ReadPrimitive<std::uint64_t>(cur, end);
    if (static_cast<std::size_t>(end - cur) < value_len) {
      throw std::runtime_error("checkpoint deserialize: value payload truncated");
    }
    out.value_map[id] = deserialize_value(cur, static_cast<std::size_t>(value_len));
    cur += value_len;
  }
  return out;
}

}  // namespace continuum::runtime
