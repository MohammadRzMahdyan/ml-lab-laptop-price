# پیش‌بینی قیمت لپ‌تاپ

پروژه‌ برای **پیش‌بینی قیمت لپ‌تاپ‌ بر اساس مشخصات فنی** با استفاده از CatBoost و Lasso Regression.

## ساختار پروژه

```
data
│   └── laptop_data.csv
notebooks/
│   └── lasso_finall_pipeline.ipynb
outputs/             # خروجی‌ها
│   ├── figures/     # نمودارها
│   └── models/      # مدل‌ ذخیره شده
transformers/        # ترنسفورمرهای سفارشی
│   └── transformers.py
requirements.txt
```

##  اجرا

```bash
pip install -r requirements.txt
jupyter notebook notebooks/lasso_finall_pipeline.ipynb
```

**ویژگی‌ها:**

* پردازش و انتخاب  مهم‌ترین ویژگی‌ها با CatBoost
* پیش‌بینی قیمت با Lasso Regression
* نمودار Learning Curve برای بررسی overfitting
