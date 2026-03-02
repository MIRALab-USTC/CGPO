import smtplib
from email.mime.text import MIMEText

# ===== 邮箱配置（修改为你的信息） =====
SMTP_SERVER = "mail.ustc.edu.cn"   # 邮件服务器，例如 QQ邮箱是 smtp.qq.com
SMTP_PORT = 465                    # SSL 端口，QQ邮箱和大多数邮箱都是 465
SMTP_USER = "xizeliang@mail.ustc.edu.cn"  # 你的邮箱
SMTP_PASS = "eU3f6Z7iTCBAVeSz"     # SMTP 授权码（不是邮箱登录密码）
TO_EMAIL = "xizeliang@mail.ustc.edu.cn" # 接收方邮箱

# ===== 邮件内容 =====
msg = MIMEText("这是一封测试邮件，说明服务器可以正常发邮件。", "plain", "utf-8")
msg["From"] = SMTP_USER
msg["To"] = TO_EMAIL
msg["Subject"] = "邮件发送测试"

try:
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, [TO_EMAIL], msg.as_string())
    print("✅ 邮件已发送成功！")
except Exception as e:
    print("❌ 邮件发送失败：", e)