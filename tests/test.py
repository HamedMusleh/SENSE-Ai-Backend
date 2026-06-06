import arabic_reshaper
from bidi.algorithm import get_display

text = "مرحبا كيفك يا أحمد"

# إصلاح اتجاه العربي
reshaped_text = arabic_reshaper.reshape(text)
fixed_text = get_display(reshaped_text)

print(fixed_text)