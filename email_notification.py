import os
import smtplib
from dotenv import load_dotenv
from email.message import EmailMessage

from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

def get_email_configuration():
    correo_remitente = os.getenv("MY_GMAIL")
    correo_destinatario = os.getenv("MY_GMAIL")
    contraseña = os.getenv("GMAIL_PASSWORD_2P")
    asunto = "Proactive Forest ha terminado"
    mensaje = "Los experimentos terminaron, el algoritmo terminó de correrse"
    return correo_remitente, correo_destinatario, contraseña, asunto, mensaje
    

def send_finish_email():
    try:
        correo_remitente, correo_destinatario, contraseña, asunto, mensaje = get_email_configuration()
    
        # Crear el objeto del mensaje
        msg = EmailMessage()
        msg['Subject'] = asunto
        msg['From'] = correo_remitente
        msg['To'] = correo_destinatario
        msg.set_content(mensaje)

        # Conectar al servidor SMTP
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(correo_remitente, contraseña)
        servidor.send_message(msg)
        servidor.quit()
        
        print('Notification sended')
    except:
        print('Conexion error, notification not send')
    

def send_finish_file(ruta_archivo: str):
    try:
        correo_remitente, correo_destinatario, contraseña, asunto, mensaje = get_email_configuration()

        # Crear el objeto del mensaje
        msg = MIMEMultipart()
        msg['From'] = correo_remitente
        msg['To'] = correo_destinatario
        msg['Subject'] = asunto

        # Agregar el mensaje
        msg.attach(MIMEText(mensaje, 'plain'))
        binario = open(ruta_archivo, 'rb')
        parte = MIMEBase('application', 'octet-stream')
        parte.set_payload(binario.read())
        encoders.encode_base64(parte)
        parte.add_header('Content-Disposition', "attachment; filename= %s" % ruta_archivo)
        msg.attach(parte)

        # Conectar al servidor SMTP
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(correo_remitente, contraseña)
        texto = msg.as_string()
        servidor.sendmail(correo_remitente, correo_destinatario, texto)
        servidor.quit()
        
        print('File sended')
    except:
        print('Conexion error, file not send')
