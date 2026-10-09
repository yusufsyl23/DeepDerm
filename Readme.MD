# DeepDerm — Yapay Zeka Destekli Uzaktan Akne Tedavi Takip Sistemi

Dermatologların akne hastalarını uzaktan takip etmesi için geliştirilmiş uçtan uca
platform. Hasta mobil uygulamadan üç açıdan yüz fotoğrafı yükler, özel eğitilmiş bir
YOLO modeli lezyonları tespit edip şiddet skoru üretir, doktor web panelinden sonuçları
inceleyip ilaç ve not yönetimi yapar.

Sistem dört çalışan servis ile bunları besleyen bir model eğitim hattından oluşur.

---

## İçindekiler

- [Mimari](#mimari)
- [Uçtan Uca Akış](#uçtan-uca-akış)
- [AI Model Eğitim Hattı](#ai-model-eğitim-hattı)
- [Servisler](#servisler)
  - [1. Frontend — Doktor Paneli](#1-frontend--doktor-paneli)
  - [2. Backend — Spring Boot API](#2-backend--spring-boot-api)
  - [3. DermAI — AI Analiz Servisi](#3-dermai--ai-analiz-servisi)
  - [4. Mobile — Hasta Uygulaması](#4-mobile--hasta-uygulaması)
- [Tüm Servisleri Başlatma](#tüm-servisleri-başlatma)
- [Teknoloji Yığını](#teknoloji-yığını)
- [Güvenlik ve Gizlilik](#güvenlik-ve-gizlilik)
- [Geliştirme Notları](#geliştirme-notları)

---

## Mimari

```
DeepDerm/
├── mobile/        React Native + Expo SDK 54   →  Hasta uygulaması (iOS)
├── frontend/      React 18 + Vite + Tailwind   →  Doktor paneli (web)
├── backend/       Spring Boot 3.3 + PostgreSQL →  Ana REST API
├── dermai-ai/     FastAPI + YOLO + MediaPipe   →  AI analiz mikro servisi
└── Yusuf/         Ultralytics YOLO11s          →  Model eğitim hattı (offline)
```

Veri akışı yönleri:

```
  [Hasta - mobil]                      [Doktor - web]
        |                                     |
        | JWT (hasta)                         | JWT (doktor)
        v                                     v
  +-------------------------------------------------+
  |         Spring Boot REST API  (:8080)           |
  |   kimlik dogrulama, hasta/ilac/not yonetimi      |
  +-------------------------------------------------+
        |                        ^
        | POST /analyze          | analiz sonucu
        v                        |
  +-------------------------------------------------+
  |         DermAI  FastAPI  (:8000)                |
  |   YOLO inferans + MediaPipe kalite kontrol       |
  +-------------------------------------------------+
        |
        v
  +-------------------------------------------------+
  |         PostgreSQL 16  (:5432)                  |
  +-------------------------------------------------+
```

### Servis Portları

| Servis | Port | Açıklama |
|--------|------|----------|
| PostgreSQL | 5432 | Veritabanı |
| Spring Boot | 8080 | Ana REST API |
| DermAI | 8000 | AI analiz mikro servisi |
| Frontend | 5173 | Doktor web paneli |
| Metro | 8081 | React Native bundler |

---

## Uçtan Uca Akış

Bir fotoğrafın hastadan doktora ulaşana kadar izlediği yol:

1. **Çekim** — Hasta mobil uygulamada kamerayı açar.
2. **Açı doğrulama** — Fotoğraf DermAI'nin `/quality/check-angle` ucuna gider;
   MediaPipe FaceLandmarker yüzün `front` / `right` / `left` açılarından hangisinde
   olduğunu belirler. Beklenen açı tutmazsa fotoğraf kabul edilmez.
3. **Üç açı tamamlanır** — Ön görünüm → sağ yanak → sol yanak sırası izlenir.
4. **Yükleme** — Üç fotoğraf Spring Boot'un `/patients/me/photos` ucuna yüklenir.
5. **Analiz tetikleme** — Spring Boot, DermAI'nin `/analyze` ucunu çağırır.
6. **Inferans** — DermAI görüntüyü 640×640 RGB'ye hazırlar, `best.pt` ile YOLO
   inferansı çalıştırır, bounding box'ları çizip annotated görüntüyü saklar.
7. **Klinik skorlama** — `clinical/postprocess.py` lezyon sayısı ve türlerinden
   şiddet skoru üretir, sonuç veritabanına yazılır.
8. **Gizlilik** — İstenirse `/quality/mask-eyes` ile fotoğraftaki gözler maskelenir.
9. **Doktor incelemesi** — Sonuç doktor panelindeki hasta detay sayfasında, önceki
   analizlerle karşılaştırmalı olarak görünür.

---

## AI Model Eğitim Hattı

`Yusuf/` klasörü, DermAI servisinde kullanılan akne lezyon tespit modelinin eğitim
çalışmasını barındırır. Üç aşamalı bir hat ve dört eğitim sürümü içerir.

### Veri Seti

Kaynak: [Roboflow — Acne Detection V1](https://universe.roboflow.com/acne-project-2auvb/acne-detection-v1)
(CC BY 4.0), YOLO formatında dışa aktarılmış.

| Sınıf | Türkçe | Train örnek sayısı | Dengesizlik |
|---|---|---|---|
| `comedone` | komedon | 829 | en nadir (7.6× az) |
| `nodule` | nodül | 6339 | en yaygın |
| `papule` | papül | 3492 | — |
| `pustule` | püstül | 4316 | — |

| Klasör | İçerik |
|---|---|
| `Yusuf/Acne-Detection-V1-1/` | Ham Roboflow çıktısı — 1419 görüntü (train/valid/test) |
| `Yusuf/islenmiş_veri/` | Aşama 01 çıktısı — 2201 görüntü, artırım sonrası |

### Aşama 01 — Veri Ön İşleme ve Artırım

`asama_01_veri.py` · `Asama 01 veri v2.py`

Veri analizinden çıkan dört bulgu bu aşamanın tasarımını belirledi:

| Bulgu | Uygulanan çözüm |
|---|---|
| Her görüntüde yalnızca tek sınıf var | Mosaic augmentasyonu |
| Komedon 7.6× az temsil ediliyor | Otomatik oversampling |
| Bounding box'ların %54'ü *small* (640 px'de ~34×34 px) | Hassas, bbox-güvenli dönüşümler |
| Görüntülerin %23'ü karanlık | Dinamik CLAHE |

Ayrıca klinik geçerlilik için cilt tonu adaptasyonu uygulanır.

### Aşama 02 — Model İnşası

`asama_02_model_insasi.py` · `asama_02_model_insasi_v2.py`

YOLOv11s üzerine üç modifikasyon:

1. **SABottleneck (Spatial Attention Bottleneck)** — FPN/neck katmanına eklenir, P3/P4/P5
   feature map'lerinde çalışır. Küçük bbox yoğunluğu (%54 small) nedeniyle kritik:
   saç ve arka plan gürültüsünü bastırıp lezyon sinyalini güçlendirir.
2. **Dinamik Focal Loss** — γ = 2.0 × √(max_count / class_count) formülüyle sınıf başına
   hesaplanır: komedon γ≈3.51, nodül γ≈1.27, papül γ≈1.71, püstül γ≈1.54. α sınıf
   ağırlıkları aynı dengesizlik oranından türetilir. Amaç objectness collapse'ı önlemek.
3. **COCO pretrained transfer learning** — `yolo11s.pt` ağırlıklarıyla başlanır; backbone
   önce dondurulur, neck + head ince ayar yapılır.

### Aşama 03 — Eğitim Döngüsü ve Sürümler

`asama_03_egitim_dongusu.py` → `_v2` → `_v3` → `_v4`

Ultralytics `DetectionTrainer` sınıfı türetilerek mosaic, multi-scale, AMP ve DDP desteği
korunur; `on_fit_epoch_end` callback'i ile ReduceLROnPlateau ve early stopping eklenir.

Ölçülen sonuçlar (en iyi epoch, doğrulama kümesi):

| Sürüm | Strateji | Epoch | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| **v2** | 640 px, SABottleneck + Focal Loss | 81 (en iyi: 66) | **0.2209** | 0.0546 | 0.286 | 0.339 |
| v3 | v2 ağırlıklarından 1024 px'e geçiş | 37 (en iyi: 17) | 0.1400 | 0.0373 | 0.163 | 0.369 |
| v4 | "Saf YOLO" — sıfırdan 1024 px | 9 (yarım kaldı) | 0.1132 | 0.0299 | 0.193 | 0.246 |

**v3 neden geriledi?** Üç sebep teşhis edildi: (a) v2'nin `best.pt` ağırlıklarıyla ani
1024 px geçişi catastrophic forgetting'e yol açtı, (b) `box=10.0` + 1024 px kombinasyonu
aşırı kutu üretip NMS time limit'e takıldı, (c) Focal Loss monkey-patch'i çekirdek
mekanizmaya müdahale ediyordu.

**v4 stratejisi** bu derslerin üzerine kuruldu: çekirdek loss'a dokunmadan, yalnızca
native parametrelerle 1024 px'in çözünürlük gücünü kullanmak.

| Parametre | v3 | v4 | Gerekçe |
|---|---|---|---|
| `base_model` | v2/best.pt | yolo11s.pt | Ağırlık şokunu önle |
| `box` | 10.0 | 7.5 | NMS time limit düzeltmesi |
| `lr0` | 0.0005 | 0.001 | Sıfırdan başlıyor |
| `mixup` | 0.15 | 0.10 | Gürültü azaltma |
| `copy_paste` | 0.25 | 0.10 | Gürültü azaltma |
| Focal Loss | γ=2.0 patch | yok | Risk minimizasyonu |
| `multi_scale` | yok | 0.5 | 512–1536 px adaptasyonu |
| ES patience | 20 | 25 | Sıfırdan yavaş öğrenir |
| Backbone çözme | epoch 10 | epoch 15 | Daha uzun warmup |

v4 koşusu 9 epoch'ta kesildiği için v2 hâlâ en iyi ölçülen sonuç. Eğitim donanımı:
RTX 5070 Ti (12.8 GB), Intel Ultra 9; batch 32, AdamW, AMP açık.

### Eğitim Çıktıları

```
Yusuf/
├── Acne-Detection-V1-1/          ham Roboflow veri seti
├── islenmiş_veri/                Aşama 01 çıktısı (artırılmış)
├── asama_01_veri.py              veri ön işleme
├── asama_02_model_insasi.py      model inşası (SABottleneck, Focal Loss)
├── asama_03_egitim_dongusu*.py   eğitim döngüsü v1-v4
├── model/, model_v2/             model tanımları (yolo11s_akne.yaml)
├── egitim_v2/                    81 epoch — en iyi sonuç
│   ├── checkpoints/              best.pt, last.pt
│   ├── logs/                     egitim_metrikleri.csv
│   └── ultralytics_runs/         results.csv, args.yaml, grafikler
├── egitim_v3/                    37 epoch — 1024 px denemesi
├── egitim_v4/                    9 epoch — saf YOLO, yarım kaldı
├── yolo11s.pt                    COCO pretrained başlangıç ağırlığı
└── yolo26n.pt                    alternatif başlangıç ağırlığı
```

> **Not:** Serviste çalışan `dermai-ai/best.pt`, `Yusuf/` altındaki eğitim çıktılarından
> ayrı bir dosyadır (hash'leri eşleşmez). Eğitim hattını bu klasörden çalıştırıp yeni bir
> ağırlık üretirseniz, servise almak için `dermai-ai/best.pt` dosyasını değiştirip
> `DERMAI_MODEL_VERSION` değerini güncellemeniz gerekir.

---

## Servisler

### 1. Frontend — Doktor Paneli

React 18 + Vite + Tailwind CSS 3 ile geliştirilmiş web uygulaması.

**Sayfalar**

| Sayfa | Yol | Açıklama |
|-------|-----|----------|
| Giriş | `/login` | JWT tabanlı doktor girişi |
| Panel | `/dashboard` | İstatistikler, son hastalar, bildirimler |
| Hastalar | `/patients` | Tüm hastalar, arama ve filtre |
| Hasta Detayı | `/patients/:id` | Fotoğraf galerisi, AI analiz sonuçları, ilaçlar, notlar, yan etkiler, acil uyarılar |
| Bildirimler | `/notifications` | Birleşik bildirim gelen kutusu |

**Yapı** — `context/AuthContext.jsx` oturum durumunu tutar, `components/ProtectedRoute.jsx`
yetkisiz erişimi engeller, `services/api.js` backend çağrılarını toplar,
`hooks/useNotifications.js` bildirim akışını yönetir.

```bash
cd frontend
npm install
npm run dev     # → http://localhost:5173
```

---

### 2. Backend — Spring Boot API

Spring Boot 3.3 + Spring Security + JWT. 45 Java sınıfı, katmanlı paket yapısı:
`controller`, `service`, `repository`, `entity`, `dto`, `security`, `config`, `exception`.

**Gereksinimler** — Java 21+, Maven 3.9+, PostgreSQL 16+

**Veri modeli** (`entity/`)

| Entity | Karşılığı |
|---|---|
| `Doctor`, `Patient` | Kullanıcı rolleri (ayrı JWT token'ları) |
| `Photo`, `AiLabel` | Yüklenen fotoğraflar ve AI tespit etiketleri |
| `Medication`, `MedicationConfirm` | Reçete edilen ilaçlar ve doz onayları |
| `DoctorNote`, `NoteRead` | Doktor notları ve okundu durumu |
| `SideEffectReport` | Hasta yan etki bildirimleri |
| `EmergencyAlert` | Acil uyarılar |

**Çevre değişkenleri**

```properties
DB_USERNAME=deepderm_user
DB_PASSWORD=yourpassword
JWT_SECRET=your-secret-key-minimum-32-characters
```

```bash
cd backend
mvn spring-boot:run     # → http://localhost:8080
```

**API Endpoint'leri**

```
# Kimlik Doğrulama
POST   /auth/login                              → Doktor JWT token
POST   /auth/patient/register                   → Hasta kaydı
POST   /auth/patient/login                      → Hasta JWT token

# Doktor → Hasta Yönetimi
GET    /patients                                → Hasta listesi
GET    /patients/{id}                           → Hasta detayı
GET    /patients/{id}/photos                    → Fotoğraflar
GET    /patients/{id}/medications                → İlaçlar
POST   /patients/{id}/medications                → İlaç ekle
GET    /patients/{id}/side-effects                → Yan etkiler
GET    /patients/{id}/emergency-alerts            → Acil uyarılar
GET    /patients/{id}/notes                      → Doktor notları
POST   /patients/{id}/notes                      → Not ekle

# Hasta (Self)
GET    /patients/me                              → Profil bilgisi
PUT    /patients/me/password                     → Şifre değiştir
GET    /patients/me/photos                       → Kendi fotoğrafları
POST   /patients/me/photos                       → Fotoğraf yükle
GET    /patients/me/medications                   → Kendi ilaçları
POST   /patients/me/medication-confirm            → İlaç dozu onayla
GET    /patients/me/doctor-notes                  → Doktor notları
GET    /patients/me/doctor-notes/unread-count     → Okunmamış not sayısı
POST   /patients/me/side-effects                  → Yan etki bildir
GET    /patients/me/emergency-alert/status        → Acil uyarı durumu
POST   /patients/me/emergency-alert               → Acil uyarı gönder

# AI Analiz
GET    /ai/analysis/{photoId}                    → Analiz sonucu
POST   /photos/{id}/analyze                      → AI analizini tetikle

# Bildirimler
GET    /notifications                            → Tüm bildirimler
PATCH  /notifications/{id}/read                  → Okundu işaretle
```

`/auth/**` dışındaki tüm uçlar için: `Authorization: Bearer <jwt_token>`

---

### 3. DermAI — AI Analiz Servisi

FastAPI tabanlı mikro servis. Spring Boot tarafından çağrılır, kendi PostgreSQL
şemasını Alembic ile yönetir.

**Modüller**

| Modül | Sorumluluk |
|---|---|
| `models/detector.py` | YOLO inferansı (`best.pt`) |
| `models/heuristic_detector.py` | Model yüklenemezse sezgisel yedek tespit |
| `quality/image_quality.py` | MediaPipe ile yüz açısı doğrulama, göz maskeleme |
| `clinical/postprocess.py` | Lezyon sayısı → klinik şiddet skoru |
| `routers/analyze.py`, `routers/quality.py` | HTTP uçları |
| `db/migrations/` | Alembic şema migration'ları |
| `training/train_yolo11.py` | Servis içi eğitim yardımcısı |

**Gereksinimler** — Python 3.10+

```bash
cd dermai-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000
```

**Çevre değişkenleri** (`dermai-ai/.env`)

```env
DEEPDERM_DB_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/deepderm
DERMAI_MODEL_PATH=best.pt            # özel eğitilmiş YOLO modeli
DERMAI_MODEL_VERSION=best-v1
DERMAI_CONF_THRESHOLD=0.25
DERMAI_IOU_THRESHOLD=0.45
DERMAI_HEURISTIC_FALLBACK=false      # true → model yoksa sezgisel tespite düşer
STORAGE_DIR=storage/annotated
CORS_ORIGINS=http://localhost:8080,http://localhost:5173
```

**Endpoint'ler**

```
GET   /health                   → Servis sağlık durumu ve model bilgisi
POST  /analyze                  → Fotoğraf analizi (YOLO inferans)
GET   /results/{photo_id}       → Analiz sonucu getir
GET   /compare                  → İki analizi karşılaştır

POST  /quality/check-angle      → Yüz açısı doğrulama (front / right / left)
POST  /quality/mask-eyes        → Göz maskeleme (gizlilik)
```

Yüz açısı doğrulaması `face_landmarker.task` (MediaPipe FaceLandmarker) ile yapılır.
Docker ile çalıştırmak için `Dockerfile` hazırdır.

---

### 4. Mobile — Hasta Uygulaması

React Native + Expo SDK 54 (TypeScript). Fiziksel iOS cihaz ve custom dev build gerektirir.

**Ekranlar** (`mobile/src/screens/`)

| Ekran | Açıklama |
|---|---|
| `LoginScreen`, `RegisterScreen` | JWT tabanlı hasta kimlik doğrulama |
| `HomeScreen` | Hızlı erişim menüsü, okunmamış not sayacı |
| `PhotoUploadScreen` | Kamera ile 3 açıdan fotoğraf çekme ve yükleme |
| `MedicationsScreen` | İlaç listesi ve doz onaylama |
| `DoctorNotesScreen` | Doktordan gelen notlar |
| `SideEffectScreen` | İlaç yan etkisi raporlama |
| `EmergencyScreen` | Doktora anlık acil uyarı |
| `ProfileScreen` | Hesap bilgileri ve şifre değiştirme |

```bash
cd mobile
npm install
bash update-ip.sh                          # ağ değişince backend IP'sini güncelle
npx expo start --clear --port 8081
npx expo run:ios --device <UDID>
```

---

## Tüm Servisleri Başlatma

```bash
# 1. PostgreSQL
LC_ALL=C /opt/homebrew/opt/postgresql@16/bin/postgres -D /opt/homebrew/var/postgresql@16 &

# 2. Spring Boot
cd backend && mvn spring-boot:run -q &

# 3. DermAI
cd dermai-ai && .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload &

# 4. Frontend
cd frontend && npm run dev &

# 5. Mobile (önce IP güncelle)
cd mobile && bash update-ip.sh
npx expo start --clear --port 8081 &
npx expo run:ios --device <UDID>
```

---

## Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| Doktor paneli | React 18, Vite, Tailwind CSS 3 |
| Hasta mobil | React Native, Expo SDK 54, Vision Camera v4, TypeScript |
| Backend API | Spring Boot 3.3, Java 21, Spring Security |
| AI servisi | FastAPI, Ultralytics YOLO, MediaPipe, OpenCV |
| Model eğitimi | Ultralytics YOLO11s, PyTorch, AdamW, AMP |
| Veritabanı | PostgreSQL 16 |
| ORM | Spring Data JPA / Hibernate · SQLAlchemy (async) |
| Migration | Flyway (backend) · Alembic (dermai-ai) |
| Kimlik doğrulama | JWT — doktor ve hasta için ayrı token'lar |
| Build | Maven, npm, Expo EAS, Docker |

---

## Güvenlik ve Gizlilik

- JWT kimlik doğrulama, doktor ve hasta için ayrı token'lar
- BCrypt şifre hash'leme (cost 12)
- Rol tabanlı erişim kontrolü (doktor / hasta)
- Fotoğraflarda göz maskeleme seçeneği
- CORS kısıtlaması
- Stateless oturum yönetimi
- KVKK uyumlu veri silme desteği

---

## Geliştirme Notları

- **Ağ değişikliği:** Geliştirme makinesi farklı bir WiFi'ye bağlandığında
  `mobile/update-ip.sh` çalıştırılmalı ve Metro yeniden başlatılmalıdır.
- **Public WiFi:** İstemci izolasyonu cihaz-cihaz iletişimini engelleyebilir;
  telefon hotspot'u kullanmak daha güvenilir.
- **iOS build:** Expo Go değil custom dev build gerekir, fiziksel cihaz şarttır.
- **Model değişimi:** `dermai-ai/best.pt` değiştirildiğinde `DERMAI_MODEL_VERSION`
  da güncellenmelidir; `/health` ucu çalışan model sürümünü döndürür.
- **Repo boyutu:** `node_modules/`, `backend/target/` ve eğitim ağırlıkları sürüm
  kontrolüne dahil edilmiş durumda; depo yaklaşık 600 MB'tır.
