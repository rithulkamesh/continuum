#pragma once

#include <string>

namespace continuum::runtime {

/// Second-stage check on a semantic-cache candidate: is it safe to serve the
/// answer cached for `cached_prompt` to `new_prompt`?
///
/// Embedding similarity measures topical closeness, not "has the same
/// answer": near-miss edits ("enable" vs "disable", "Australia" vs "Austria",
/// "5" vs "50") score as high as paraphrases. A verifier runs only on
/// candidates that already cleared the similarity threshold. Implement it in
/// C++ or subclass `continuum._native.HitVerifier` in Python (e.g. an LLM
/// judge).
class HitVerifier {
 public:
  virtual ~HitVerifier() = default;
  /// True to serve the cached answer, false to treat the lookup as a miss.
  virtual bool verify(const std::string& cached_prompt, const std::string& new_prompt,
                      float similarity) const = 0;
  /// Stable name, for metrics and logs.
  virtual std::string name() const = 0;
};

/// Default verifier: rejects candidates that are a *minimal edit* of the
/// cached prompt, the signature of a near-miss, while letting rewordings
/// through.
///
/// Prompts are lowercased and tokenized; stopwords are dropped and simple
/// suffixes normalized. The candidate is rejected when:
/// - the numbers differ (digits or number words: "5" vs "50", "two" vs "four");
/// - the content words are the same but a polarity word is swapped
///   ("to"/"from", "on"/"off", "with"/"without") or a negation appears on one
///   side only;
/// - the same content words appear in a different order around a directional
///   word ("convert 10 miles to km" vs "convert 10 km to miles");
/// - aligning the content words leaves only substitutions of at most two
///   words each ("reset my password" vs "reset my username"), with no
///   insertions or deletions and at least one word in common.
///
/// It is conservative by design: a reworded paraphrase that happens to
/// differ by a single synonym swap ("cancel" vs "end") is also rejected,
/// costing a cache miss rather than serving a wrong answer. Measured in
/// benchmarks/reports/semantic-false-hits.md.
class LexicalNearMissVerifier final : public HitVerifier {
 public:
  struct Verdict {
    bool accept = true;
    std::string reason;  ///< "numbers", "polarity", "negation", "direction", "substitution", "ok", ...
  };
  static Verdict explain(const std::string& cached_prompt, const std::string& new_prompt);
  bool verify(const std::string& cached_prompt, const std::string& new_prompt, float similarity) const override;
  std::string name() const override { return "lexical-near-miss-v1"; }
};

}  // namespace continuum::runtime
