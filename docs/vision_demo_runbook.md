# LR-YOLO Vision Lab — chạy demo trên Kaggle

## File cần dùng

Upload `notebooks/demo/01_kaggle_lryolo_demo.ipynb` lên Kaggle. Notebook nhúng
frontend đã build, backend và helper/project source; không cần push GitHub hoặc
cài Node trên Kaggle để chạy bản đã đóng gói. Không sửa notebook Drill.

Attach wheel MMCV T4 đã dùng và output model. Với mỗi model, giữ:
- Checkpoint tin cậy (`best_*.pth` để demo model được chọn).
- `resolved_config.py` cùng run, `provenance.json`, `session_state.json`.
- Không cần cache prototype, audit report hoặc checkpoint clean để suy luận S3.
- Bản mới có sẵn 24 kết quả đánh giá từ log đã cung cấp; có thể thay REPORTS.

Chọn Python3.10.10 / GPU T4 / Internet. Giữ chế độ token Ngrok đã cấu hình;
chỉ dùng Kaggle Secrets nếu chọn chế độ `secrets`.
Không cần OPENAI_API_KEY, GITHUB_TOKEN hoặc dataset của Drill. Server dùng1GPU;
chạy demo trong session riêng để tránh giành VRAM với training.

## Các cell notebook

1. Giải nén bundle có hash.
2. Cell tham số: chọn gallery low_light/haze, pool, registry và output.
3. Setup stack OpenMMLab đã có + serving packages còn thiếu. Có guard Python,
   Torch/MMCV; không chạy lại installer thủ công khi stack đang hoạt động.
4. Dò checkpoint, chuẩn bị dataset/gallery và ghi config runtime.
5. Khởi động FastAPI, health check và mở link Ngrok.
6. Cuối notebook có cell dừng đúng server/tunnel này; không xóa output.

Nếu dùng chế độ secrets, bật quyền truy cập secret trong giao diện Kaggle.
Token nằm phía notebook/server, không đưa vào frontend hoặc log. Link chỉ hoạt động khi session
Kaggle và tunnel còn sống. Giới hạn Ngrok tùy tài khoản, không hứa24/7.

## Dataset và nút random

Mặc định `GALLERY_FAMILIES=['low_light','haze']`, `GALLERY_POOL='clean_val'`,
`AUTO_DOWNLOAD_DATASETS=True`. Nếu thiếu, helper tải bundle clean VOC gốc của
tác giả, giải nén an toàn và dùng manifest validation sạch trong bundle để
random. Nó KHÔNG chia lại dữ liệu hay ghi vào dataset gốc. Bundle có cả train,
nhưng mặc định random chỉ từ clean internal val. Cả hai miền được chuẩn bị sẵn.
Có thể giữ một family trong danh sách nếu muốn giảm dung lượng tải trên Kaggle.
Dataset lớn KHÔNG được tải local khi build.

Random VOC_Haze hoạt động ngay khi chọn Haze trên giao diện; lớp haze5 và
low-light10 khác nhau. Nếu dataset đã mount, dùng DATA_OVERRIDES hoặc để helper dò native
layout. `GALLERY_POOL='fixed_test'` dùng bộ đã degradation offline; UI mặc định
không thêm gamma/haze. Random là minh họa, không dùng để chọn checkpoint hoặc
đổi hyperparameter theo final test. Model VOC không có semantics BDD10classes.

Nút random lấy ảnh thật từ manifest, không sinh ảnh giả. Nếu pool hiện chỉ có
một ảnh hoặc ảnh trùng lần trước, vẫn là random sampling với replacement.

## Chọn checkpoint

`MODELS=None` dò các sidecar của main train run trong `/kaggle/input`. Chỉ đọc
metadata nhỏ khi dò, không torch.load mọi checkpoint. Chọn một best checkpoint
trong từng run; nhiều file best sẽ được báo để chọn tường minh. Nếu chưa có
adaptation model, clean model vẫn có thể dùng thử nhưng hiển thị là Clean.
Không gắn nhãn clean thành kết quả LR-YOLO dark đã train.

Có thể khai báo MODELS tường minh:

```python
MODELS = [dict(
    id='v10-lowlight-s1', label='YOLOv10-S · S1',
    architecture='v10', degradation='low_light', stage='dark', case_id=6,
    checkpoint='/kaggle/input/THU_MUC_RUN/best_voc_standard_mAP_epoch_N.pth',
    config='/kaggle/input/THU_MUC_RUN/resolved_config.py',
)]
```

Thay bằng path thật. Baseline có entry tương tự, case_id0. Load strict toàn bộ
weights, kiểm tra stage/case, kiến trúc và class order. Trong serving, selective
model dùng native full reconstruction prediction có cùng tensors; không chạy
sampling/loss khi inference. Bỏ random-dark/haze preprocessor khỏi inference.
Không upload checkpoint qua giao diện web. Torch pickle/config chỉ từ nguồn
server do chủ project chọn tin cậy.

Một model cache trênGPU. So sánh A/B chạy tuần tự cùng degraded input; cold load
có thể chậm, đặc biệt khi đọc checkpoint trên Kaggle Input. Cache Torch/VGG cho
LPIPS có thể cần download lần đầu theo behavior nguồn; lỗi được trả về job và
server.log. Không âm thầm bỏ weights/loss module chỉ để load được.

## Chức năng giao diện

- Upload JPG/PNG/WebP hoặc random VOC; giữ tỉ lệ ảnh.
- Degradation bật tắt rõ ràng, có preview/slider trước-sau. Gamma1.5..5, haze
  beta.05...14/A.5 theo source. Haze radial depth tính trên virtual640canvas để
  ảnh4K không tạo negative-depth; đây là transform demo offline, không phải
  tái tạo bit-identical training float sau letterbox. Ảnh cuối lưuPNG và dùng
  chung cho mọi model trong request.
- Bounding boxes, class/confidence, bộ lọc, ngưỡng, zoom/pan đồng bộ và fullscreen.
- So sánh hai model cùng family; v8/v10 được ghi rõ kiến trúc. Nhiều box hoặc
  confidence lớn không chứng minh tốt hơn. Không ground truth thì không AP.
- GT nét đứt từ XML gốc của gallery; rawVOC convention được khai báo, không coi
  đây là parser/evaluator benchmark. Ảnh upload không tự có ground truth.
- Reconstruction tùy chọn là RGB nhánh phụ, không phải detector cascade.
- Export PNG dùng threshold/class hiện tại; JSON giữ toàn bộ native predictions,
  hash input/checkpoint, settings và timing scope. Chưa có artifact nào trước
  khi request hoàn tất. Model latency gồm native preprocessing/postprocessing,
  CUDA sync, sau1warmup; không gồm network/coldload. Không gọi là benchmarkFPS.
- Trang thực nghiệm đọc REPORTS, không chạy full evaluator. Source, actual
  epochs, VRAM scope hiển thị; dữ liệu thiếu không được tạo giả.
- Minh họa LR-YOLO gốc, S1, S2, S3 từ gallery annotated trên virtual640canvas, seed
  cố định. Đây là mô phỏng sampling, không chạy optimizer, không phải attention.

Giới hạn bản đầu:20MB/16MP, tối đa2model/request cùngfamily,4job chờ/chạy,
100job/session. Không tự xóa artifacts để tránh mất export; session dài nên
export rồi khởi động server mới. Không video/webcam/live training ở bản đầu.

## Phát triển local và rebuild

Frontend (dùng dependencies đã có): `cd vision_demo/web`, `npm run build`. Sau đó ở Git repo root `LR-YOLO/`:
`python notebooks/demo/build_notebook.py`. Bundle nhúng đúng dist; thay source
phải build lại để Kaggle nhận code mới.

Backend cần serving packages và OpenMMLab khi infer. Khi chạy manual, đặt
PYTHONPATH gồm repoLR-YOLO và `notebooks/demo/support/` (shared checkpoint loader), đặt
LRYOLO_MODELS/GALLERY/REPORTS trỏ JSON và LRYOLO_REPO trỏ repo chứa sampler.
Chạy `python -m uvicorn vision_demo.server.app:app --host 127.0.0.1 --port 8501 --workers 1`.

Tests API/image fixtures dùng fake engine chỉ trong test; ứng dụng không có
mock-inference mode. Local validation không chứng minh checkpoint/GPU/Ngrok đã
hoạt động; cần smoke ảnh thật trên Kaggle và đối chiếu native inference trước
khi bảo vệ. Trạng thái test thực tế được ghi trong docs/experiment_log.md.


## Kaggle Secrets trả404 / ConnectionError

Nếu lỗi tại UserSecretsClient.get_secret, lỗi xảy ra trước khi start server,
chưa chứng minh Ngrok token hoặc checkpoint sai. Kiểm tra Add-ons > Secrets,
label chính xác NGROK_AUTH_TOKEN và quyền cho notebook hiện tại; kiểm tra Internet.
Nếu cấu hình đã đúng mà backend Secrets vẫn lỗi, trong phiên interactive có thể
chạy cell riêng (không ghi token vào source/output):

```python
import os
from getpass import getpass
os.environ['NGROK_AUTHTOKEN'] = getpass('Ngrok authtoken: ')
```

Rerun cell server của notebook mới; helper ưu tiên biến môi trường trên trước
Kaggle Secrets. Không tự prompt trong Save & Run All. Với notebook cũ gọi trực
tiếp UserSecretsClient, thay dòng ngrok.set_auth_token đó bằng
`ngrok.set_auth_token(os.environ['NGROK_AUTHTOKEN'])` trong phiên đang mở.

## Tải checkpoint tác giả (2026-11-16)

Notebook mới mặc định:

```python
AUTHOR_RELEASES = ['voc_dark_v8', 'voc_dark_v10', 'voc_haze_v8', 'voc_haze_v10']
AUTHOR_CHECKPOINT_MEMBERS = {}
AUTHOR_CACHE = Path('/kaggle/working/lryolo_author_models')
MODELS = None
# Giữ NGROK_TOKEN_MODE đang có trong notebook của bạn.
```

Không cần có checkpoint tự train để bắt đầu. Cell chuẩn bị registry sẽ tải từ
link chính thức rồi thêm model có nhãn Author. Chọn prompt khi chạy interactive;
ô nhập token được ẩn, không lưu token vào notebook. Với Save & Run All, dùng
NGROK_TOKEN_MODE='secrets' hoặc cấu hình môi trường riêng thay vì prompt.

Bốn ID được xác minh từ README chính thức:

| ID | Model | Classes |
|---|---|---:|
| voc_dark_v10 | VOC_Dark / LR-YOLOv10-S |10|
| voc_dark_v8 | VOC_Dark / LR-YOLOv8-S |10|
| voc_haze_v10 | VOC_Haze / LR-YOLOv10-S |5|
| voc_haze_v8 | VOC_Haze / LR-YOLOv8-S |5|

Ví dụ tải cả hai bản low-light:
`AUTHOR_RELEASES=['voc_dark_v10','voc_dark_v8']`. Chỉ tải model trong danh sách;
mặc định mới chọn cả bốn để demo đủ low-light/haze và v8/v10. Muốn chỉ dùng
model tự train, đặt AUTHOR_RELEASES=[] và khai báo MODELS hoặc dùng discovery.

Release cache gồm file download, author_checkpoint.pth và release.json chứa
nguồn, file/member, hashes và config. Đã có cache thì không tải lại. Không giải
nén toàn archive hoặc chạy config lấy từ archive. Chỉ dùng native eval config
trong repo đã pin. Checkpoint tự train bổ sung vào registry mà không thay model
Author. Các AP paper không được tự thêm vào bảng như kết quả đo của demo.

Một release có nhiều checkpoint phù hợp sẽ yêu cầu chọn EXACT member bằng
AUTHOR_CHECKPOINT_MEMBERS, không tự chọn theo AP. VOC_Dark/v10 dùng filename
best_pascal_voc_mAP_epoch_97.pth đã có bằng chứng lịch sử. Ba release còn lại
được xác minh link/config, chưa giải nén real archive local; xử lý unique best
hoặc một checkpoint duy nhất, và báo rõ nếu layout khác. Nếu Drive quota hoặc
trả HTML, quá trình sẽ báo lỗi; không tạo registry giả hay bỏ qua kiểm tra.
Inference strict-check keys/shapes/class metadata khi model được load lần đầu;
nguồn công bố không chứng minh toàn bộ lịch sử training của checkpoint.

Nguồn: https://github.com/diasuki/LR-YOLO#pretrained-models . Không tải release,
chạy inference hay validate GPU thật trong bước cập nhật local này; tests dùng
archive fixtures để kiểm tra selection/cache và config cho cả bốn ID.


### Phiên bản notebook riêng đã cấu hình token

Theo yêu cầu chủ project, bản notebook được tạo tại workspace này đã chọn
NGROK_TOKEN_MODE='inline' và có token trong cell tham số: không hiện prompt và
không gọi Kaggle Secrets. Builder đọc token từ vision_demo/runtime/ngrok_token.txt
(file local được ignore, không nằm trong source bundle) để giữ cấu hình khi
rebuild. Clone không có file cấu hình này sẽ mặc định prompt; có thể chọn lại
mode bằng cell tham số. Không in giá trị token trong log.


## Ghi chú hỗ trợ case 8 trước đây (tên hiện tại: S2)

The demo now accepts case8 (coverage-guided object/uniform). S7 remains reserved.
Native inference drops the training-only coverage_candidates knob along with
other sampler fields and strict-loads the original prediction tensors. The
illustration tab uses its actual pure-Python geometry sampler; it does not
perform eight network predictions. Current UI labels and default comparison
selection are described in the 2026-11-20 section below.


## Bản demo S1 / S2 / S3 — 2026-11-20

Tên trong giao diện thống nhất với slides; mã checkpoint và contract giữ nguyên:

| Giao diện | case_id trong registry | Cơ chế khi huấn luyện |
|---|---:|---|
| LR-YOLO gốc | 0 | Tái tạo toàn ảnh |
| S1 | 6 | Một vùng theo đối tượng + một vùng ngẫu nhiên |
| S2 | 8 | Chọn vùng theo độ bao phủ, giữ ngân sách hai vùng |
| S3 | 9 | S2 + prototype sạch có trọng số độ tin cậy |

Các case cũ 1–5 vẫn đọc được và ghi rõ là thử nghiệm cũ; không đổi tên chúng
thành S1/S2/S3. Clean checkpoint được ghi riêng là Clean.
Chọn kiến trúc YOLOv8-S hoặc YOLOv10-S ở thanh phía trên. So sánh A/B dùng cùng
kiến trúc đang chọn và cùng ảnh đầu vào. Checkpoint chưa mount không được tạo
thay thế; cần attach output dark đã train của phương án muốn demo.

S3 chỉ dùng prototype trong train. Serving chuyển constructor sang detector
native có cùng tensors, bỏ đúng các option train của prototype/crop, dùng tiền
xử lý đánh giá thông thường và vẫn `load_state_dict(strict=True)`. Không sửa
checkpoint, không bỏ qua weights thiếu/thừa, không đọc lại cache prototype.
Model architecture/class order và checkpoint hash vẫn được kiểm tra.

Trang thực nghiệm có bốn thẻ phương pháp, hình chọn vùng có cùng seed giữa S1/S2
và hai tỷ lệ diện tích LPIPS 32%, L1 18%. Đây không phải FLOPs toàn mạng. S3 dùng
cùng hình vùng cắt S2; sơ đồ bên dưới giải thích loss bổ sung, không giả lập
prototype similarity từ ảnh hoặc chạy loss trong inference.

`config/research_results.json` chép 24 bản ghi từ hai CSV kết quả đã đối chiếu
trong project (haze/lowlight, v8/v10, S1/S2/S3, hai domain mỗi nhánh).
Giữ giá trị `map_raw` và `all_objects_raw`, tên file bằng chứng và nguồn log.
Không suy ra epoch, tổng thời gian hoặc VRAM nếu log kết quả không cung cấp.
Không tự gộp chi phí đo một epoch vào thời gian full training.

Notebook có thay đổi ở:

- **Cell 3** (cấu hình, `demo-002`): `REPORTS=None` dùng số liệu đóng gói;
  `REPORTS=[]` ẩn bảng; có thể truyền danh sách riêng như cũ. Các tùy chọn
  checkpoint, token, gallery và author release giữ cách dùng cũ.
- **Cell 5** (chuẩn bị registry/gallery, `demo-004`): đọc kết quả đóng gói nếu
  `REPORTS is None`. Tự dò các output train có sidecar, bao gồm case 9.
- **Cell 2** (giải nén, `demo-001`): payload mới chứa source, frontend và số liệu.

Mặc định mới chuẩn bị cả hai miền: `GALLERY_FAMILIES=['low_light','haze']`
và cả bốn baseline tác giả trong `AUTHOR_RELEASES`. Giảm hai danh sách này nếu
chỉ cần một miền hoặc một kiến trúc. S1/S2/S3 vẫn cần mount output dark tương ứng.

Cần upload notebook mới để nhận payload mới. Chỉ đổi CASE_ID trong notebook demo
cũ không bổ sung code serving S3 hay giao diện mới. Nếu thay payload trong một
phiên đã import app cũ, dừng server và restart kernel để tránh module cache cũ.


## Haze có đủ tính năng như low-light — 2026-11-20

Cell3 mặc định bật cả hai gallery và cả bốn checkpoint tác giả; không đổi cách
nạp checkpoint tự train hoặc token. Chuyển sang Haze trên giao diện để dùng:
ảnh VOC_Haze ngẫu nhiên, upload ảnh, bật/tắt sương mù và chỉnh beta .05–.14,
preview trước/sau, bounding box/class/confidence, GT từ XML của gallery,
reconstruction phụ, so sánh hai model cùng đầu vào và export PNG/JSON.
Minh họa S1/S2/S3 và bảng12 kết quả haze có sẵn giống low-light.

Sương mù thật hoặc gallery fixed_test giữ degradation tắt mặc định; không cần
tạo thêm sương mù. Model haze dùng5 lớp gốc; không dùng checkpoint low-light
10lớp thay cho haze. S3 inference vẫn không cần cache prototype.
