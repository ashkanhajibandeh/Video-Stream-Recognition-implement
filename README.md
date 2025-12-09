# Video-Stream-Recognition-implement
پیاده سازی شناسایی جریان ویدیو با استفاده از شکل بیت استریم برای QoE شبكه موبايل
 تولید دیتاست مصنوعی

سه کلاس مثل مقاله:

Label	نوع ترافیک	توضیح
0	Video-HAS	رفتار chunk-based با پیک‌های دوره‌ای
1	Other Apps	رفتار burst-y و غیر پیوسته
2	Start-Screen / Background	ترافیک بسیار کم و پراکنده
 تولید Downlink و Uplink جدا

(مطابق مقاله فقط Down بیشتر معنی‌دار است ولی Uplink هم برای مدل لحاظ می‌شود)

 ساخت window-ها با طول 60 / 300 / 600 ثانیه

هر نمونه = یک پنجره مستقل → آماده برای Training

 حذف پنجره‌های multi-app

 خروجی مرحله 1
دیتاست	طول پنجره	نمونه‌های ایجاد شده
X60.npy	60s	~1k
X300.npy	300s	~600
X600.npy	600s	~300

این دیتاست دقیقاً ورودی مرحله 2 است که در آن PyTorch-Dataset، DataLoader و مدل CNN اضافه می‌کنیم.

در این مرحله هدف فقط سه چیز است:

 آنچه در این فاز انجام می‌دهیم
جزء	توضیح
BitstreamDataset	کلاس PyTorch Dataset که همان X,Y مرحله اول را می‌خواند
DataLoader	مینی‌بچ‌ها را برای مدل آماده می‌کند
معماری CNN پایه مطابق مقاله	Conv1D چند لایه با Average Pooling انتهایی — دو کانال Downlink+Uplink
 کد کامل مرحله 2

(فقط DataLoader + CNN — هیچ Train/Evalی هنوز اجرا نمی‌شود)

نکته مهم: این کد مستقیماً روی خروجی مرحله 1 سوار می‌شود.

 خروجی این مرحله
نتیجه	آماده؟
دیتاست قابل استفاده PyTorch	
DataLoader برای 60/300/600 ثانیه	
مدل Conv1D دوکاناله (Downlink+Uplink)
 مرحله بعد

 خروجی این مرحله:
قابلیت	انجام می‌شود؟
Train / Validation Split	 از داده‌های مرحله ۲
Training Loop + Progress Bar	 با TQDM
ذخیره بهترین مدل (best_model.pth)	✔ بر اساس Validation Accuracy
ذخیره کامل متریک‌ها در .npy + نمودار	✔ برای گزارش‌دهی
بدون کامنت در کد (طبق درخواست تو)	✔✔✔


 پس از اجرای این مرحله، فایل‌های زیر ساخته می‌شود:
فایل	کاربرد
best_model.pth	بهترین وزن ذخیره‌شده مدل CNN
history.npy	تاریخچه train/val (برای رسم در مرحله ۴)
accuracy_curve.png	نمودار دقت در طول epochs (معادل نتایج مقاله)
report.txt	Precision / Recall / F1 گزارش آماده ارائه


اسکریپت کامل PyTorch (یک فایل video_recognition.py)

این اسکریپت همه‌چیز را (تولید دیتاست مصنوعی، windowing، Dataset/Dataloader، مدل، آموزش، ارزیابی، گراف‌ها و ذخیره خروجی‌ها) انجام می‌دهد.



