import subprocess
import smtplib
from email.mime.text import MIMEText
import time

# ===== 邮箱配置（修改为你的信息） =====
SMTP_SERVER = "mail.ustc.edu.cn"   # 邮件服务器，例如 QQ邮箱是 smtp.qq.com
SMTP_PORT = 465                    # SSL 端口，QQ邮箱和大多数邮箱都是 465
SMTP_USER = "xizeliang@mail.ustc.edu.cn"  # 你的邮箱
SMTP_PASS = "eU3f6Z7iTCBAVeSz"     # SMTP 授权码（不是邮箱登录密码）
TO_EMAIL = "xizeliang@mail.ustc.edu.cn" # 接收方邮箱

# ===== 检测 GPU 是否空闲 =====
def gpu_has_free():
    """返回 True 如果有任意一张 GPU 显存为 0"""
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    mem_list = [int(x) for x in result.decode().strip().split("\n")]
    print("当前 GPU 显存占用:", mem_list)
    return any(mem == 0 for mem in mem_list)

# ===== 发送邮件 =====
def send_email(subject, content):
    msg = MIMEText(content, "plain", "utf-8")
    msg["From"] = SMTP_USER
    msg["To"] = TO_EMAIL
    msg["Subject"] = subject
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, [TO_EMAIL], msg.as_string())
    print(f"✅ 已发送邮件: {subject}")

# ===== 主循环 =====
if __name__ == "__main__":
    while True:
        if gpu_has_free():
            # 第一次发现空闲
            send_email("GPU 空闲提醒", "有空闲 GPU 了！快去占用！")
            print("进入二次检测阶段（10分钟，每30秒一次）...")
            free_all_the_time = True  # 标记10分钟内是否一直空闲

            for _ in range(20):  # 10分钟 / 每30秒一次 = 20次
                time.sleep(30)
                if not gpu_has_free():  # 如果被占满了
                    free_all_the_time = False
                    send_email("下一个程序已启动", "GPU 又被占满了，下一个程序已启动！")
                    break

            if free_all_the_time:
                send_email("实验完成提醒", "这次是真的跑完实验了！")
                break  # 结束脚本
            else:
                print("重新进入长期检测模式...")
                # 不 break，继续长期检测
        time.sleep(30)