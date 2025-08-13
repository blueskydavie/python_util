import pyautogui
import time
import os
import pyperclip

# 启动应用
os.startfile("C:\Program Files (x86)\Internet Explorer\iexplore.exe")
time.sleep(5)  # 等待应用加载

# 点击“上传”按钮（需要你提前知道坐标）
pyautogui.click(x=1476, y=130)
pyautogui.alert(text='input url', title='提示', button='OK')
time.sleep(1)

# 输入文件路径
pyautogui.write("https://*******/login")
pyautogui.press("enter")
pyautogui.press("enter")
# 增加等待时间确保页面完全加载
time.sleep(5)

# 点击用户名输入框并输入
try:
    pyautogui.click(x=1681, y=1073)
    time.sleep(1)
    pyperclip.copy("kevin")  # 写入剪贴板
    pyautogui.hotkey("ctrl", "v")  # 粘贴
    # pyautogui.write("付文超")
except Exception as e:
    print(f"输入用户名失败: {e}")

# 点击密码输入框并输入
try:
    pyautogui.click(x=1686, y=1195)
    time.sleep(1)
    pyperclip.copy("密码")  # 写入剪贴板
    pyautogui.hotkey("ctrl", "v")  # 粘贴
    # pyautogui.write("密码")
except Exception as e:
    print(f"输入密码失败: {e}")

    

# 添加调试信息
print("用户名和密码输入完成")
pyautogui.click(x=1900, y=1427)

pyautogui.press("enter")
pyautogui.press("enter")

time.sleep(3)

pyautogui.click(x=77, y=587)

time.sleep(3)

pyautogui.click(x=464, y=1331)

time.sleep(3)

pyautogui.click(x=475, y=1516)

time.sleep(3)

pyautogui.write("H:\images\test.jpg")

# 点击“处理”按钮
# pyautogui.click(x=120, y=300)
