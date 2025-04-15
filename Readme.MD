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
- ✅ **Green Zone**: No human detected in front of door.  
- ❌ **Red Zone**: Human enters defined visual box (danger/presence alert).  
- ✅ Reverts to **Green** once the person exits.  

---

## 🤖 Core Tech Used  
- **YOLOv8 (Ultralytics)** for object detection  
- **OpenCV** for visual processing  
- **Numpy**, **Matplotlib** for visual support  
- **Future Hardware Integration**:  
  - Buzzer / Alert system connected via camera module or Raspberry Pi/Arduino

---

## 📷 Demo & Results
- 🟢 No Human in Door Range
Description: **Visual box is green, indicating safety.**

<p align="center">
  <img src="https://github.com/Programmer-Rakesh/AI_based_Home_Security_system/blob/main/Demo_media/Safe.png" width="550" height="200">
</p>

---

- 🔴 Human in Door Range
Description: **Visual box turns red as human enters the zone.**

<p align="center">
  <img src="https://github.com/Programmer-Rakesh/AI_based_Home_Security_system/blob/main/Demo_media/Alert.png" width="550" height="200">
</p>

## 🧠 Libraries & Packages  

```python
import cv2  
import pickle
import numpy as np    
from ultralytics import YOLO
import matplotlib.pyplot as plt
from skimage.transform import resize

