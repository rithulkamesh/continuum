#include <continuum/runtime/hit_verifier.hpp>

#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace continuum::runtime {
namespace {

const std::unordered_set<std::string>& Stopwords() {
  static const std::unordered_set<std::string> kStop = {
      "a", "an", "the", "i", "me", "my", "you", "your", "we", "our", "it", "its", "is", "are", "was",
      "were", "be", "been", "do", "does", "did", "how", "what", "which", "who", "whom", "when", "where",
      "why", "can", "could", "should", "would", "will", "shall", "might", "must", "of", "at", "for", "by",
      "about", "and", "or", "but", "if", "then", "so", "that", "this", "these", "those", "there", "here",
      "as", "have", "has", "had", "get", "got", "want", "need", "please", "tell", "show", "give", "just",
      "than", "too", "very", "many", "much", "some", "any", "all", "also", "like", "let", "im"};
  return kStop;
}

const std::unordered_set<std::string>& PolarityWords() {
  static const std::unordered_set<std::string> kPolar = {
      "to", "from", "into", "onto", "on", "off", "in", "out", "up", "down", "before", "after", "above",
      "below", "over", "under", "not", "no", "never", "without", "with"};
  return kPolar;
}

const std::unordered_set<std::string>& NumberWords() {
  static const std::unordered_set<std::string> kNum = {
      "zero", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
      "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "million"};
  return kNum;
}

bool IsDigits(const std::string& w) {
  return !w.empty() && std::all_of(w.begin(), w.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Lowercase, drop possessive "'s", expand "n't" to " not", split on non [a-z0-9].
std::vector<std::string> Tokenize(const std::string& text) {
  std::string s;
  s.reserve(text.size() + 8);
  for (char c : text) s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  std::string cleaned;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\'' && i + 1 < s.size() && s[i + 1] == 's' &&
        (i + 2 == s.size() || !std::isalnum(static_cast<unsigned char>(s[i + 2])))) {
      ++i;  // skip "'s"
      continue;
    }
    if (s[i] == 'n' && i + 2 < s.size() && s[i + 1] == '\'' && s[i + 2] == 't') {
      cleaned += " not";
      i += 2;
      continue;
    }
    cleaned.push_back(s[i]);
  }
  std::vector<std::string> out;
  std::string cur;
  for (char c : cleaned) {
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
      cur.push_back(c);
    } else if (!cur.empty()) {
      out.push_back(std::move(cur));
      cur.clear();
    }
  }
  if (!cur.empty()) out.push_back(std::move(cur));
  return out;
}

std::string Normalize(const std::string& w) {
  for (const char* suf : {"ing", "ed", "es"}) {
    const std::string sfx(suf);
    if (EndsWith(w, sfx) && w.size() - sfx.size() >= 4) return w.substr(0, w.size() - sfx.size());
  }
  if (EndsWith(w, "s") && !EndsWith(w, "ss") && w.size() - 1 >= 3) return w.substr(0, w.size() - 1);
  return w;
}

// Word forms sharing a 6-letter prefix ("summarize" / "summary") count as one word.
bool SameWord(const std::string& a, const std::string& b) {
  return a == b || (a.size() >= 6 && b.size() >= 6 && a.compare(0, 6, b, 0, 6) == 0);
}

struct Parsed {
  std::vector<std::string> content;
  std::multiset<std::string> numbers;
  std::set<std::string> polarity;
};

Parsed Parse(const std::string& text) {
  Parsed p;
  for (const auto& t : Tokenize(text)) {
    if (IsDigits(t) || NumberWords().count(t) != 0) {
      p.numbers.insert(t);
    } else if (PolarityWords().count(t) != 0) {
      p.polarity.insert(t);
    } else if (Stopwords().count(t) == 0) {
      p.content.push_back(Normalize(t));
    }
  }
  return p;
}

std::set<std::string> Minus(const std::set<std::string>& a, const std::set<std::string>& b) {
  std::set<std::string> out;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::inserter(out, out.begin()));
  return out;
}

// True when aligning a and b (longest common subsequence) leaves only
// replacement gaps of at most two words per side and at least one match:
// the shape of a minimal edit rather than a rewording.
bool IsMinimalSubstitution(const std::vector<std::string>& a, const std::vector<std::string>& b) {
  const std::size_t n = a.size();
  const std::size_t m = b.size();
  std::vector<std::vector<int>> lcs(n + 1, std::vector<int>(m + 1, 0));
  for (std::size_t i = n; i-- > 0;) {
    for (std::size_t j = m; j-- > 0;) {
      lcs[i][j] = a[i] == b[j] ? lcs[i + 1][j + 1] + 1 : std::max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  if (lcs[0][0] == 0) return false;
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t gap_a = 0;
  std::size_t gap_b = 0;
  auto close_gap = [&]() {
    if (gap_a == 0 && gap_b == 0) return true;
    const bool ok = gap_a > 0 && gap_b > 0 && gap_a <= 2 && gap_b <= 2;
    gap_a = gap_b = 0;
    return ok;
  };
  while (i < n && j < m) {
    if (a[i] == b[j]) {
      if (!close_gap()) return false;
      ++i;
      ++j;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      ++gap_a;
      ++i;
    } else {
      ++gap_b;
      ++j;
    }
  }
  gap_a += n - i;
  gap_b += m - j;
  return close_gap();
}

}  // namespace

LexicalNearMissVerifier::Verdict LexicalNearMissVerifier::explain(const std::string& cached_prompt,
                                                                  const std::string& new_prompt) {
  Parsed a = Parse(cached_prompt);
  Parsed b = Parse(new_prompt);
  if (a.numbers != b.numbers) return {false, "numbers"};

  // Canonicalize prefix-equal word forms so the alignment treats them as one word.
  std::vector<std::string> canon;
  auto canonical = [&](const std::string& w) {
    for (const auto& c : canon) {
      if (SameWord(w, c)) return c;
    }
    canon.push_back(w);
    return w;
  };
  for (auto& w : a.content) w = canonical(w);
  for (auto& w : b.content) w = canonical(w);

  auto sorted_a = a.content;
  auto sorted_b = b.content;
  std::sort(sorted_a.begin(), sorted_a.end());
  std::sort(sorted_b.begin(), sorted_b.end());
  const auto only_a = Minus(a.polarity, b.polarity);
  const auto only_b = Minus(b.polarity, a.polarity);
  if (sorted_a == sorted_b) {
    if (a.content != b.content) {
      for (const char* dir : {"to", "from", "into", "than"}) {
        if (a.polarity.count(dir) != 0 && b.polarity.count(dir) != 0) return {false, "direction"};
      }
      return {true, "reordered"};
    }
    if (!only_a.empty() && !only_b.empty()) return {false, "polarity"};
    for (const char* neg : {"not", "no", "never", "without"}) {
      if ((a.polarity.count(neg) != 0) != (b.polarity.count(neg) != 0)) return {false, "negation"};
    }
    return {true, "same"};
  }
  if (IsMinimalSubstitution(a.content, b.content)) return {false, "substitution"};
  return {true, "ok"};
}

bool LexicalNearMissVerifier::verify(const std::string& cached_prompt, const std::string& new_prompt,
                                     float /*similarity*/) const {
  return explain(cached_prompt, new_prompt).accept;
}

}  // namespace continuum::runtime
