# 🛡️ Smart Security Zone – Visual Door Alert System 🚪  
**By First Year B.Tech ECE Student – Rakesh Roy**

---

## 📌 Problem Statement  
- Traditional doorbell cams lack **active visual alerts**.  
- No real-time **zone detection** or visual indicators.  
- Security often fails in detecting human presence before engagement.  

> *"What if your door could visually signal danger before you even open it?"*

---

## 💡 Our Solution: **Smart Security Zone**  
An **AI-based object detection system** that creates a **virtual box** outside a house door.  

### 🚦 Visual Alert Logic  
- ✅ **Green Zone**: No human detected in front of the door.  
- ❌ **Red Zone**: Human enters the defined visual box (danger/presence alert).  
- ✅ Reverts to **Green** once the person exits.  

---

## 🤖 Core Tech Used  
- **YOLOv8 (Ultralytics)** for object detection  
- **OpenCV** for visual processing  
- **NumPy**, **Matplotlib** for visual support  
- **Future Hardware Integration**:  
  - Buzzer / Alert system connected via camera module or Raspberry Pi/Arduino

---

## 🧠 Libraries & Packages  

```python
import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
from ultralytics import YOLO  
