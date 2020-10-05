import cv2
import numpy as np

#pra capturar o vídeo
cap = cv2.VideoCapture(0)

#carregando o modelo pré-treinado
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
	#pra ler a captura de vídeo
  ret,frame = cap.read()
  #convertendo a captura pra preto e branco pra ficar mais facil na hora de virar binário
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #modelando o classificador de rosto
  faces = faceClassif.detectMultiScale(gray, 1.3, 5)

  #capturar o retangulo do rosto
  for (x,y,w,h) in faces:
    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
  #pra mostrar a gravação  
  cv2.imshow('frame',frame)
  

  #botão pra parar a cam
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()