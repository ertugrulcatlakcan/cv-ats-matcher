# CV Analiz ve Eşleştirme Sistemi (ATS) — Faz 1: Veri Alımı, OCR ve Metin Çıkarma

Tamamen lokal ve açık kaynak Python 3.12 kütüphaneleriyle çalışan (harici/ücretli LLM API'si kullanmayan) modüler CV okuma, görsel ön işleme, akıllı yönlendirme (Smart Router), OCR, metin temizleme ve doğrulama (benchmark) altyapısı.

> **Kapsam:** Bu depo şu an **Faz 1** (Ingestion, OCR, Cleaning ve Ground Truth Benchmark) katmanını içerir. Faz 2'de eklenecek TF-IDF eşleştirme, spaCy NLP ve Streamlit arayüzü, `src/models.py` içindeki `ExtractionResult` veri sözleşmesine doğrudan bağlanacak şekilde tasarlanmıştır.

## Mimari ve Dizin Yapısı

```text
.
├── .streamlit/
│   └── config.toml          # Kurumsal koyu tema ve sunucu yapılandırması
├── app.py                   # Kurumsal koyu temalı Streamlit web arayüzü (Faz 1.5)
├── main.py                  # Tek dosya veya klasör işleyen argparse CLI giriş noktası
├── requirements.txt         # Python 3.12 bağımlılıkları
├── pytest.ini               # Pytest yapılandırması
├── src/
│   ├── __init__.py          # Genel paket export'ları
│   ├── config.py            # pathlib yolları, sabitler (MIN_TEXT_LENGTH=50, OCR_DPI=300, RANDOM_SEED=42)
│   ├── models.py            # ExtractionResult (Faz 2 sözleşmesi) ve özel exception sınıfları
│   ├── preprocessor.py      # OpenCV tabanlı Türkçe-yol güvenli ön işleme (grayscale, denoise, koşullu deskew)
│   ├── extractor.py         # PyMuPDF tabanlı Smart Router (.pdf, .png/.jpg/.jpeg, .docx)
│   ├── cleaner.py           # Teknik terimleri koruyan regex temizleyici (C++, C#, .NET, Node.js, CI/CD)
│   ├── evaluator.py         # Resume.csv Ground Truth kıyaslamalı OCR stres testi ve jiwer WER/CER benchmark
│   └── ui_helpers.py        # Streamlit'ten bağımsız, test edilebilir UI yardımcı katmanı
└── tests/                   # Her modül ve UI yardımcı katmanı için pytest testleri
```

## Kurulum

### 1. Tesseract OCR (v5.x) Kurulumu

- **Windows:**
  - [UB-Mannheim Tesseract 5 Installer](https://github.com/UB-Mannheim/tesseract/wiki) ile kurun veya `winget install UB-Mannheim.TesseractOCR` komutunu çalıştırın.
  - `C:\Program Files\Tesseract-OCR` klasörünü sistem `PATH` değişkenine ekleyin (PATH'te değilse `src/config.py` standart Windows kurulum dizinine otomatik fallback yapar veya `TESSERACT_CMD` ortam değişkenini okur).
- **macOS (Homebrew):**
  ```bash
  brew install tesseract
  ```
- **Linux (Ubuntu / Debian):**
  ```bash
  sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
  ```

Doğrulama:
```bash
tesseract --version
```

### 2. Python 3.12 Sanal Ortam (`.venv`) Kurulumu

- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
- **macOS / Linux:**
  ```bash
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

## Kullanım

### Web Arayüzü (Streamlit UI — Faz 1.5)

Kurumsal koyu temalı web arayüzünü `.venv` içinde başlatmak için:
```bash
streamlit run app.py
```

Öne çıkan arayüz özellikleri:
- **Eklemeli Çalışma Alanı (Additive Upload Queue):** Önceki yüklenen CV'leri sıfırlamadan yeni dosyalar ekler, SHA-256 içerik özetiyle kopyaları tekilleştirir.
- **Uyarlanabilir Düzen:** 1 dosyada doğrudan KPI + Yan Yana İnceleme (Split View), 2+ dosyada Toplu Özet Tablosu + toplu JSON/CSV dışa aktarma moduna geçer.
- **Original ↔ Preprocessed (OpenCV) Önizleme:** PDF sayfaları ve görsellerde orijinal görüntü ile Tesseract 5'e giren ön işlenmiş OpenCV görüntüsü arasında anlık geçiş sunar.
- **OCR Benchmark & Faz 2 İskeleti:** `outputs/ocr_benchmark.csv` metriklerini grafik/tablo olarak görselleştirir ve Faz 2 `Job Matching` sekme iskeletini hazır sunar.

### CLI (`main.py`)

Tek bir CV dosyasını özet tablo olarak işleme:
```bash
python main.py path/to/resume.pdf
```

JSON çıktısı alma ve UTF-8 dosyaya kaydetme:
```bash
python main.py path/to/resume.pdf --json --output outputs/result.json
```

Bir klasördeki tüm desteklenen CV'leri (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.docx`) işleme:
```bash
python main.py path/to/cv_folder --recursive --json
```

Karanlık veya gölgeli taramalar için opsiyonel adaptif eşiklemeyi açma:
```bash
python main.py path/to/photo_cv.jpg --threshold --dpi 300
```

### OCR Stres Testi ve Benchmark (`src/evaluator.py`)

`dataset/Resume.csv` ve `dataset/data/INFORMATION-TECHNOLOGY/` altındaki PDF'ler üzerinde metin katmanını kaldırarak (yarısı temiz, yarısı 2–3° döndürülmüş + Gaussian gürültülü) adil `jiwer` WER/CER testi çalıştırır ve `outputs/ocr_benchmark.csv` dosyasını üretir:

```bash
python -m src.evaluator --sample-size 10 --seed 42
```

Opsiyonel argümanlar: `--pdf-dir`, `--csv-path`, `--output`, `--sample-size`, `--seed`, `--dpi`, `-v`.

## Faz 2 Veri Sözleşmesi (`ExtractionResult`)

`src/models.py` içindeki `ExtractionResult` dataclass'ı Faz 2 NLP/eşleştirme katmanı için sabit şema sağlar:

| Alan | Tip | Açıklama |
| --- | --- | --- |
| `file_path` | `str` | Çözümlenmiş kaynak dosya yolu |
| `filename` | `str` | Dosya adı |
| `method` | `Literal["direct_pdf", "ocr_pdf", "ocr_image", "docx"]` | Kullanılan çıkarma yöntemi |
| `page_count` | `int` | İşlenen sayfa sayısı |
| `raw_text` | `str` | Temizlenmemiş ham metin |
| `cleaned_text` | `str` | Teknik terimler korunarak temizlenmiş metin |
| `processing_time_sec` | `float` | İşlem süresi (saniye) |
| `warnings` | `list[str]` | Ölümcül olmayan uyarılar |

## Testler

Tüm birim ve entegrasyon testlerini `.venv` içinde çalıştırmak için:
```bash
pytest -v
```
