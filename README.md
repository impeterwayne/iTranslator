**1. Cài đặt Python**

[Download Python | Python.org](https://www.python.org/downloads/)

**2. Cài đặt dependencies**

``` 
pip install -r requirements.txt 
```

**3. Định nghĩa tên dòng app để LLM dịch chuẩn hơn trong file ```.env```**

```
APP_NAME=Pregnancy Tracker and Calculator
```

**4. Tiến hành dịch**

- Thay thế file ```strings.xml``` tại thư mục iTranslator

- Mở CMD tại thư mục iTranslator chạy

```
python main.py
```
	
Với file dài thì chạy
	
```
python chunk.py
```