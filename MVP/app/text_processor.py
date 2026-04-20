import re
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class TurkishSoundex:
    """Turkish-adapted Soundex algorithm for phonetic error detection"""

    TURKISH_CHAR_MAP = {
        "a": "A",
        "e": "E",
        "ı": "I",
        "i": "I",
        "o": "O",
        "ö": "O",
        "u": "U",
        "ü": "U",
        "b": "B",
        "c": "C",
        "ç": "C",
        "d": "D",
        "f": "F",
        "g": "G",
        "ğ": "G",
        "h": "H",
        "j": "J",
        "k": "K",
        "l": "L",
        "m": "M",
        "n": "N",
        "p": "P",
        "r": "R",
        "s": "S",
        "ş": "S",
        "t": "T",
        "v": "V",
        "w": "W",
        "x": "X",
        "y": "Y",
        "z": "Z",
    }

    CODE_MAP = {
        "A": "0",
        "E": "0",
        "I": "1",
        "O": "2",
        "U": "2",
        "B": "3",
        "C": "4",
        "D": "5",
        "F": "6",
        "G": "7",
        "H": "8",
        "J": "9",
        "K": "0",
        "L": "1",
        "M": "2",
        "N": "3",
        "P": "4",
        "R": "5",
        "S": "6",
        "T": "7",
        "V": "8",
        "W": "9",
        "X": "0",
        "Y": "1",
        "Z": "2",
    }

    VOWELS = set("aeıioöuuAEIOÖUÜ")

    def encode(self, word: str) -> str:
        """Encode a word to its Turkish Soundex representation"""
        if not word or not word.strip():
            return ""

        word = word.lower().strip()
        word = self._normalize_turkish(word)

        if not word:
            return ""

        first_letter = word[0]
        encoded = first_letter.upper()

        prev_code = self.CODE_MAP.get(
            self.TURKISH_CHAR_MAP.get(first_letter, first_letter.upper()), ""
        )

        for char in word[1:]:
            turkish_char = self.TURKISH_CHAR_MAP.get(char, char.upper())
            code = self.CODE_MAP.get(turkish_char, "")

            if code and code != prev_code:
                encoded += code

            if char not in self.VOWELS:
                prev_code = code

        return encoded[:4].ljust(4, "0") if len(encoded) < 4 else encoded[:4]

    def _normalize_turkish(self, word: str) -> str:
        """Normalize Turkish characters"""
        replacements = {
            "ç": "c",
            "Ç": "C",
            "ğ": "g",
            "Ğ": "G",
            "ı": "i",
            "İ": "I",
            "ö": "o",
            "Ö": "O",
            "ş": "s",
            "Ş": "S",
            "ü": "u",
            "Ü": "U",
        }
        for old, new in replacements.items():
            word = word.replace(old, new)
        return word

    def get_phonetic_key(self, word: str) -> str:
        """Get phonetic key for comparison"""
        return self.encode(word)


class VisualSimilarityChecker:
    """Detects visual letter confusion errors common in dyslexia"""

    VISUAL_PAIRS = {
        "b": ["d"],
        "d": ["b"],
        "p": ["q", "b"],
        "q": ["p"],
        "b": ["p", "d"],
        "m": ["n"],
        "n": ["m"],
        "u": ["ü"],
        "ü": ["u"],
        "o": ["ö"],
        "ö": ["o"],
        "a": ["e"],
        "e": ["a"],
        "i": ["ı"],
        "ı": ["i"],
        "f": ["t"],
        "t": ["f"],
        "g": ["j"],
        "j": ["g"],
        "s": ["ş", "z"],
        "ş": ["s"],
        "z": ["s"],
        "c": ["ç"],
        "ç": ["c"],
    }

    CONFUSION_WEIGHTS = {
        ("b", "d"): 1.0,
        ("d", "b"): 1.0,
        ("p", "q"): 1.0,
        ("q", "p"): 1.0,
        ("p", "b"): 0.8,
        ("b", "p"): 0.8,
        ("m", "n"): 0.9,
        ("n", "m"): 0.9,
        ("u", "ü"): 0.7,
        ("ü", "u"): 0.7,
        ("o", "ö"): 0.7,
        ("ö", "o"): 0.7,
    }

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def weighted_levenshtein(self, s1: str, s2: str) -> float:
        """Calculate weighted Levenshtein distance with visual confusion penalties"""
        if len(s1) < len(s2):
            return self.weighted_levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = [0.0] * (len(s2) + 1)

        for j in range(len(s2) + 1):
            previous_row[j] = j * 0.5

        for i, c1 in enumerate(s1):
            current_row = [(i + 1) * 0.5]

            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1.0
                deletions = current_row[j] + 1.0

                weight = self.CONFUSION_WEIGHTS.get((c1.lower(), c2.lower()), 1.0)
                if c1.lower() != c2.lower() and c1.lower() in self.VISUAL_PAIRS:
                    weight = self.CONFUSION_WEIGHTS.get((c1.lower(), c2.lower()), 0.3)

                substitutions = previous_row[j] + (
                    0.0 if c1.lower() == c2.lower() else weight
                )
                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    def find_visual_errors(self, word: str, reference_word: str) -> Dict:
        """Find visual similarity errors between two words"""
        distance = self.weighted_levenshtein(word.lower(), reference_word.lower())

        errors = []
        for i, (c1, c2) in enumerate(zip(word.lower(), reference_word.lower())):
            if c1 != c2:
                if c2 in self.VISUAL_PAIRS.get(c1, []):
                    errors.append(
                        {
                            "position": i,
                            "typed": c1,
                            "expected": c2,
                            "type": "visual_confusion",
                        }
                    )

        return {
            "distance": distance,
            "normalized_distance": distance / max(len(reference_word), 1),
            "error_count": len(errors),
            "errors": errors,
        }

    def get_most_similar_word(
        self, word: str, word_list: List[str]
    ) -> Optional[Tuple[str, float]]:
        """Find the most similar word from a list"""
        if not word_list:
            return None

        word = word.lower()
        best_match = None
        best_distance = float("inf")

        for ref_word in word_list:
            ref = ref_word.lower()
            if len(ref) == 0:
                continue

            distance = self.weighted_levenshtein(word, ref)
            if distance < best_distance:
                best_distance = distance
                best_match = ref_word

        return (best_match, best_distance) if best_match else None


class TurkishSyllableSplitter:
    """Turkish syllable analyzer for split/fusion detection"""

    VOWELS = "aeıioöuuAEIOÖUÜ"
    VOWEL_GROUPS = [
        "ai",
        "ei",
        "ıi",
        "oi",
        "öi",
        "ui",
        "üe",
        "ao",
        "eo",
        "ıo",
        "uo",
        "üe",
    ]

    def split_syllables(self, word: str) -> List[str]:
        """Split a Turkish word into syllables"""
        word = word.lower().strip()
        if not word:
            return []

        syllables = []
        current = ""

        for i, char in enumerate(word):
            current += char

            if char in self.VOWELS:
                if i + 1 < len(word) and word[i + 1] in self.VOWELS:
                    continue

                if (
                    i + 2 < len(word)
                    and word[i + 1] not in self.VOWELS
                    and word[i + 2] in self.VOWELS
                ):
                    pass
                else:
                    syllables.append(current)
                    current = ""

        if current:
            syllables.append(current)

        return syllables

    def count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        return len(self.split_syllables(word))

    def detect_split_errors(self, text: str) -> Dict:
        """Detect potential syllable split errors"""
        words = text.split()
        split_errors = []

        common_splits = {
            "kel": "em",
            "ok": "ul",
            "hi": "ka",
            "ye": "ye",
            "ba": "şka",
            "da": "ha",
            "gi": "bi",
            "ni": "ni",
        }

        for word in words:
            word_lower = word.lower()
            for error, correct in common_splits.items():
                if error in word_lower:
                    split_errors.append(
                        {
                            "word": word,
                            "suspected_error": error + correct,
                            "type": "possible_split",
                        }
                    )

        return {
            "split_error_count": len(split_errors),
            "split_errors": split_errors[:5],
        }


class TurkishWordValidator:
    """Turkish word dictionary validator"""

    def __init__(self):
        self.common_words = self._load_basic_dictionary()
        self.soundex = TurkishSoundex()
        self._build_soundex_index()

    def _load_basic_dictionary(self) -> set:
        """Load basic Turkish word dictionary"""
        words = {
            "ve",
            "ile",
            "için",
            "ama",
            "çünkü",
            "ya",
            "da",
            "de",
            "ki",
            "ne",
            "bu",
            "o",
            "biz",
            "sen",
            "onlar",
            "bizim",
            "sizin",
            "onların",
            "bu",
            "şu",
            "o",
            "bir",
            "iki",
            "üç",
            "dört",
            "beş",
            "altı",
            "yedi",
            "sekiz",
            "dokuz",
            "on",
            "yirmi",
            "otuz",
            "kırk",
            "elli",
            "altmış",
            "yetmiş",
            "seksen",
            "doksan",
            "yüz",
            "bin",
            "milyon",
            "milyar",
            "ev",
            "okul",
            "kitap",
            "kalem",
            "silgi",
            "defter",
            "masa",
            "sandalye",
            "kapı",
            "pencere",
            "duvar",
            "çatı",
            "baca",
            "komşu",
            "aile",
            "anne",
            "baba",
            "çocuk",
            "kardeş",
            "dede",
            "nine",
            "amca",
            "teyze",
            "dayı",
            "hala",
            "bebek",
            "dost",
            "araba",
            "bisiklet",
            "tren",
            "uçak",
            "gemi",
            "otobüs",
            "tren",
            "metro",
            "taksi",
            "su",
            "ekmek",
            "et",
            "tavuk",
            "balık",
            "peynir",
            "yumurta",
            "süt",
            "çay",
            "kahve",
            "mektup",
            "telefon",
            "bilgisayar",
            "televizyon",
            "radyo",
            "gazete",
            "dergi",
            "kitap",
            "roman",
            "şiir",
            "hikaye",
            "masal",
            "fabl",
            "oyun",
            "film",
            "müzik",
            "resim",
            "spor",
            "futbol",
            "basketbol",
            "voleybol",
            "tenis",
            "yüzme",
            "koşu",
            "yürüyüş",
            "dağ",
            "deniz",
            "göl",
            "nehir",
            "orman",
            "çayır",
            "tarla",
            "bağ",
            "bahçe",
            "çiçek",
            "ağaç",
            " yaprak",
            "dal",
            "gövde",
            "kök",
            "meyve",
            "sebze",
            "elma",
            "armut",
            "muz",
            "portakal",
            "limon",
            "üzüm",
            "karpuz",
            "kavun",
            "çilek",
            "domates",
            "biber",
            "salatalık",
            "soğan",
            "sarımsak",
            "patates",
            "havuç",
            "güneş",
            "ay",
            "yıldız",
            "bulut",
            "rüzgar",
            "yağmur",
            "kar",
            "dolu",
            "gökkuşağı",
            "kış",
            "yaz",
            "ilkbahar",
            "sonbahar",
            "pazartesi",
            "salı",
            "çarşamba",
            "perşembe",
            "cuma",
            "cumartesi",
            "pazar",
            "bugün",
            "yarın",
            "dün",
            "bugün",
            "şimdi",
            "sonra",
            "önce",
            "her zaman",
            "hiç",
            "bazen",
            "sık sık",
            "çok",
            "az",
            "biraz",
            "tamamen",
            "hiçbir",
            "bazı",
            "her",
            "herkes",
            "hiçkimse",
            "kimse",
            "başkası",
            "başka",
            "ben",
            "sen",
            "o",
            "biz",
            "siz",
            "onlar",
            "benim",
            "senin",
            "onun",
            "bizim",
            "sizin",
            "onların",
            "bu",
            "şu",
            "o",
            "şu",
            "bu",
            "mavi",
            "kırmızı",
            "yeşil",
            "sarı",
            "siyah",
            "beyaz",
            "turuncu",
            "mor",
            "pembemsi",
            "kahverengi",
            "gri",
            "büyük",
            "küçük",
            "orta",
            "geniş",
            "dar",
            "uzun",
            "kısa",
            "kalın",
            "ince",
            "ağır",
            "hafif",
            "sıcak",
            "soğuk",
            "ılık",
            "yumuşak",
            "sert",
            "taze",
            "bayat",
            "tatlı",
            "ekşi",
            "tuzlu",
            "acı",
            "iyi",
            "kötü",
            "güzel",
            "çirkin",
            "doğru",
            "yanlış",
            "yeni",
            "eski",
            "genç",
            "yaşlı",
            "zengin",
            "fakir",
            "aç",
            "tok",
            "hast",
            "sağlıklı",
            "yorgun",
            "dinç",
            "mutlu",
            "üzgün",
            "kızgın",
            "sakin",
            "heyecanlı",
            "korkmuş",
            "şaşkın",
            "meraklı",
            "çalışkan",
            "tembel",
            "savurgan",
            "cimri",
            "dürüst",
            "yalancı",
            "akıllı",
            "aptal",
            "zeki",
            "salak",
            "hakkında",
            "önünde",
            "arkasında",
            "yanında",
            "altında",
            "üstünde",
            "arasında",
            "içinde",
            "dışında",
            "üzerinde",
            "sayesinde",
            "yüzünden",
            "karşı",
            "göre",
            "göre",
            "kadar",
            "gibi",
            "ile",
            "veya",
            "fakat",
            "lakin",
            "halbuki",
            "oysaki",
            "demek",
            "ki",
            "de",
            "da",
            "mi",
            "mı",
            "mu",
            "mü",
            "hani",
            "yani",
            "işte",
            "evet",
            "hayır",
            "peki",
            "tamam",
            "olur",
            "sağol",
            "teşekkürler",
            "rica ederim",
            "afiyet olsun",
            "hoşça kal",
            "görüşürüz",
            "merhaba",
            "selam",
            "alo",
            "tam",
            "yarım",
            "çeyrek",
            "bütün",
            "tamamen",
            "kısmen",
            "hiç",
            "neredeyse",
            "sanki",
            "belki",
            "muhtemelen",
            "muhakkak",
            "kesinlikle",
            "elbette",
            "tabii",
            "tabi",
            "evvela",
            "sonra",
            "önce",
            "beraber",
            "birlikte",
            "ayrı",
            "farklı",
            "aynı",
            "benzer",
            "diğer",
            "öbür",
            "sıradan",
            "özel",
            "genel",
            "karma",
            "saf",
            "temiz",
            "kirli",
            "boş",
            "dolu",
            "dolu",
            "yok",
            "var",
            "olmak",
            "yapmak",
            "vermek",
            "almak",
            "gelmek",
            "gitmek",
            "düşünmek",
            "bilmek",
            "istemek",
            "görmek",
            "duymak",
            "söylemek",
            "yazmak",
            "okumak",
            "öğrenmek",
            "öğretmek",
            "anlamak",
            "hatırlamak",
            "unutmak",
            "başlamak",
            "bitirmek",
            "devam etmek",
            "durmak",
            "kalkmak",
            "oturmak",
            "yatmamak",
            "uynamak",
            "yürümek",
            "koşmak",
            "uçmak",
            "yüzmek",
            "çıkmak",
            "inmek",
            "geçmek",
            "dönmek",
            "gelmek",
            "varmak",
            "yetmek",
            "acıkmak",
            "susamak",
            "uyumak",
            "uyanmak",
            "gülmek",
            "ağlamak",
            "korkmak",
            "sevmek",
            "beğenmek",
            "saymak",
            "hesaplamak",
            "çözmek",
            "bulmak",
            "aratmak",
            "bulmak",
            "vermek",
            "almak",
            "taşımak",
            "kaldırmak",
            "indirmek",
            "açmak",
            "kapatmak",
            "çevirmek",
            "döndürmek",
            "itmek",
            "çekmek",
            "atma",
            "tutma",
            "bırakmak",
            "yakalamak",
            "vurmak",
            "dövmek",
            "kesmek",
            "biçmek",
            "delmek",
            "çizmek",
            "boyamak",
            "silmek",
            "yıkamak",
            "temizlemek",
            "ütülemek",
            "ütü",
            "pişirmek",
            "hazırlamak",
            "yemek",
            "içmek",
            " içecek",
            " kokmak",
            "tutmak",
            "taşımak",
            "koymak",
            "almak",
            "bırakmak",
            "çıkarmak",
            "takmak",
            "giymek",
            "soyunmak",
            "giyinmek",
            "beklemek",
            "hızla",
            "yavaş",
            "hızlı",
            "yavaşça",
            "dikkatli",
            "özenle",
            "sabırla",
            "heyecanla",
            "korkuyla",
            "sevinçle",
            "üzüntüyle",
            "öfkeyle",
            "sakinle",
            "net",
            "belirgin",
            "görünür",
            "gizli",
            "açık",
            "kapalı",
            "kilitli",
            "serbest",
            "bağlı",
            "bağımsız",
            "mümkün",
            "imkansız",
            "gerekli",
            "gereksiz",
            "önemli",
            " önemsiz",
            "faydalı",
            "zararlı",
            "güvenli",
            "tehlikeli",
        }
        return words

    def _build_soundex_index(self):
        """Build Soundex index for phonetic matching"""
        self.soundex_index = {}
        for word in self.common_words:
            key = self.soundex.get_phonetic_key(word)
            if key not in self.soundex_index:
                self.soundex_index[key] = []
            self.soundex_index[key].append(word)

    def is_valid_word(self, word: str) -> bool:
        """Check if word is in dictionary"""
        return word.lower().strip() in self.common_words

    def find_phonetic_matches(self, word: str) -> List[str]:
        """Find words with similar phonetic encoding"""
        word = word.lower().strip()
        
        try:
            key = self.soundex.get_phonetic_key(word)
        except Exception:
            return []
        
        if not key or not isinstance(key, str):
            return []

        matches = []
        for i in range(len(key)):
            try:
                partial_key = key[: i + 1] + key[i + 1 :].replace("0", "0")
                if partial_key in self.soundex_index:
                    matches.extend(self.soundex_index[partial_key])
            except Exception:
                continue

        return list(set(matches))[:10]

    def suggest_correction(self, word: str) -> Optional[str]:
        """Suggest correction for a possibly misspelled word"""
        word = word.lower().strip()

        if word in self.common_words:
            return None

        if len(word) < 2:
            return None

        matches = self.find_phonetic_matches(word)
        if not matches:
            return None

        visual_checker = VisualSimilarityChecker()

        best_match = None
        best_score = float("inf")

        for match in matches:
            distance = visual_checker.weighted_levenshtein(word, match)
            if distance < best_score:
                best_score = distance
                best_match = match

        if best_match and best_score < 3:
            return best_match

        return None


class DyslexiaTextMetrics:
    """Main class combining all dyslexia-specific text metrics"""

    def __init__(self):
        self.soundex = TurkishSoundex()
        self.visual_checker = VisualSimilarityChecker()
        self.syllable_splitter = TurkishSyllableSplitter()
        self.validator = TurkishWordValidator()

    def analyze(self, text: str) -> Dict:
        """Analyze text for dyslexia-specific indicators"""
        words = self._extract_words(text)

        visual_metrics = self._analyze_visual_similarity(text, words)
        phonetic_metrics = self._analyze_phonetic_errors(text, words)
        syllable_metrics = self._analyze_syllables(text)
        structure_metrics = self._analyze_structure(text)

        metrics = {
            "visual": visual_metrics,
            "phonetic": phonetic_metrics,
            "syllable": syllable_metrics,
            "structure": structure_metrics,
        }

        risk_score = self._calculate_risk_score(metrics)

        return {
            "metrics": metrics,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
        }

    def _extract_words(self, text: str) -> List[str]:
        """Extract clean words from text"""
        words = re.findall(r"[a-zA-ZığüşöçİĞÜŞÖÇ]+", text.lower())
        return [w.strip() for w in words if len(w) > 0]

    def _analyze_visual_similarity(self, text: str, words: List[str]) -> Dict:
        """Analyze visual letter confusion errors"""
        if not words:
            return {"error_count": 0, "error_rate": 0, "errors": []}

        visual_errors = []

        for word in words:
            if len(word) < 2:
                continue

            for i, char in enumerate(word):
                if char in self.visual_checker.VISUAL_PAIRS:
                    for j, ref_char in enumerate(word):
                        if i != j and ref_char in self.visual_checker.VISUAL_PAIRS.get(
                            char, []
                        ):
                            visual_errors.append(
                                {
                                    "word": word,
                                    "position": i,
                                    "confused_chars": f"{char}/{ref_char}",
                                    "type": "visual",
                                }
                            )

        error_count = len(visual_errors)

        return {
            "error_count": error_count,
            "error_rate": error_count / len(words) if words else 0,
            "errors": visual_errors[:10],
        }

    def _analyze_phonetic_errors(self, text: str, words: List[str]) -> Dict:
        """Analyze phonetic spelling errors"""
        if not words:
            return {"error_count": 0, "error_rate": 0, "errors": []}

        phonetic_errors = []

        for word in words:
            if len(word) < 2:
                continue

            suggestion = self.validator.suggest_correction(word)
            if suggestion:
                phonetic_errors.append(
                    {"word": word, "suggestion": suggestion, "type": "phonetic"}
                )

        error_count = len(phonetic_errors)

        return {
            "error_count": error_count,
            "error_rate": error_count / len(words) if words else 0,
            "errors": phonetic_errors[:10],
        }

    def _analyze_syllables(self, text: str) -> Dict:
        """Analyze syllable structure"""
        words = text.split()

        syllable_counts = []
        for word in words:
            clean_word = re.sub(r"[^a-zA-ZığüşöçİĞÜŞÖÇ]", "", word)
            if clean_word:
                count = self.syllable_splitter.count_syllables(clean_word)
                syllable_counts.append(count)

        if not syllable_counts:
            return {"avg_syllables": 0, "multi_syllable_count": 0, "errors": []}

        multi_syllable = sum(1 for c in syllable_counts if c > 3)

        return {
            "avg_syllables": sum(syllable_counts) / len(syllable_counts),
            "multi_syllable_count": multi_syllable,
            "multi_syllable_rate": multi_syllable / len(syllable_counts),
            "max_syllables": max(syllable_counts) if syllable_counts else 0,
        }

    def _analyze_structure(self, text: str) -> Dict:
        """Analyze word structure errors (fusion/split)"""
        words = text.split()

        word_lengths = [len(w) for w in words]

        fusion_candidates = []
        split_candidates = []

        for word in words:
            clean = re.sub(r"[^a-zA-ZığüşöçİĞÜŞÖÇ]", "", word)

            if len(clean) > 10:
                common_suffixes = [
                    "lerin",
                    "ların",
                    "nın",
                    "nin",
                    "da",
                    "de",
                    "den",
                    "dan",
                ]
                for suffix in common_suffixes:
                    if clean.endswith(suffix[:-1]) and len(clean) > len(suffix) + 3:
                        fusion_candidates.append(word)
                        break

            if len(clean) < 4 and len(clean) > 0:
                if clean in [
                    "da",
                    "de",
                    "mi",
                    "mı",
                    "mu",
                    "mü",
                    "ye",
                    "ni",
                    "na",
                    "ne",
                ]:
                    split_candidates.append(word)

        word_counts = Counter(word_lengths)
        unusual_lengths = sum(1 for c in word_counts if c < 3 or c > 15)

        return {
            "word_count": len(words),
            "avg_word_length": sum(word_lengths) / len(word_lengths) if words else 0,
            "short_word_count": sum(1 for l in word_lengths if l < 3),
            "long_word_count": sum(1 for l in word_lengths if l > 10),
            "fusion_candidates": len(fusion_candidates),
            "split_candidates": len(split_candidates),
            "unusual_length_count": unusual_lengths,
        }

    def _calculate_risk_score(self, metrics: Dict) -> float:
        """Calculate overall dyslexia risk score"""
        score = 0.0

        visual = metrics.get("visual", {})
        phonetic = metrics.get("phonetic", {})
        structure = metrics.get("structure", {})

        score += visual.get("error_rate", 0) * 2.0

        score += phonetic.get("error_rate", 0) * 2.5

        fusion_rate = structure.get("fusion_candidates", 0) / max(
            structure.get("word_count", 1), 1
        )
        score += fusion_rate * 1.5

        split_rate = structure.get("split_candidates", 0) / max(
            structure.get("word_count", 1), 1
        )
        score += split_rate * 1.0

        unusual_rate = structure.get("unusual_length_count", 0) / max(
            structure.get("word_count", 1), 1
        )
        score += unusual_rate * 0.5

        return min(score, 1.0)

    def _get_risk_level(self, score: float) -> str:
        """Determine risk level from score"""
        if score < 0.2:
            return "low"
        elif score < 0.5:
            return "medium"
        else:
            return "high"


def process_text(text: str) -> Dict:
    """Process text and return dyslexia metrics"""
    analyzer = DyslexiaTextMetrics()
    result = analyzer.analyze(text)
    return result
