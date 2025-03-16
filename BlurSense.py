
import os
import sys
import subprocess
import urllib.request
import logging
import argparse
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s")

# Configurações e constantes
REQUIRED_PACKAGES = [
    "opencv-python",
    "torch",
    "torchvision",
    "matplotlib",
    "numpy",
]
YOLO_CFG_URL = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg?raw=true"
YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3-tiny.weights"
COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
YOLO_CFG_PATH = "yolov3-tiny.cfg"
YOLO_WEIGHTS_PATH = "yolov3-tiny.weights"
COCO_NAMES_PATH = "coco.names"

BUFFER_SIZE = 100
OUTPUT_DIR = "treinamento"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "modelo_treinado.pth")
GRAPH_SAVE_PATH = os.path.join(OUTPUT_DIR, "grafico.png")

# Classes válidas para a detecção (nomes conforme coco.names)
# Para esse exemplo, além de "cup" e "person", damos tratamento especial a "cell phone".
VALID_CLASSES = {"cell phone", "cup", "person"}

# Funções para verificação e instalação de pacotes
def is_package_installed(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "show", package],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"{package} já está instalado.")
        return True
    except subprocess.CalledProcessError:
        return False

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logging.info(f"{package} instalado com sucesso!")
    except subprocess.CalledProcessError as e:
        logging.error(f"Falha ao instalar {package}: {e}")

def check_and_install_packages():
    logging.info("Verificando pacotes necessários...\n")
    for package in REQUIRED_PACKAGES:
        if not is_package_installed(package):
            logging.info(f"{package} não encontrado. Instalando...")
            install_package(package)

# Funções para download dos arquivos do YOLO
def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            logging.info(f"Baixando {filename}...")
            urllib.request.urlretrieve(url, filename)
            logging.info(f"{filename} baixado com sucesso!")
        except Exception as e:
            logging.error(f"Erro ao baixar {filename}: {e}")
    else:
        logging.info(f"{filename} já existe.")

def download_required_files():
    download_file(YOLO_CFG_URL, YOLO_CFG_PATH)
    download_file(YOLO_WEIGHTS_URL, YOLO_WEIGHTS_PATH)
    download_file(COCO_NAMES_URL, COCO_NAMES_PATH)

def check_yolo_files():
    if os.path.exists(YOLO_CFG_PATH) and os.path.exists(YOLO_WEIGHTS_PATH) and os.path.exists(COCO_NAMES_PATH):
        logging.info("Arquivos YOLO encontrados e configurados corretamente.")
    else:
        logging.error("Arquivos YOLO ausentes ou mal configurados. Realizando download...")
        download_required_files()

# Definição de uma Rede Neural Convolucional simples para aprendizado
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # Verificação e instalação de pacotes
    check_and_install_packages()

    # Verificação e download dos arquivos do YOLO
    check_yolo_files()

    # Criar pasta para salvar o treinamento
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Carregar rede YOLO com OpenCV
    try:
        logging.info("Carregando rede YOLO...")
        yolo_net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CFG_PATH)
    except Exception as e:
        logging.error("Falha ao carregar rede YOLO: %s", e)
        sys.exit(1)

    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers().flatten()]
    
    # Carregar classes do YOLO
    try:
        with open(COCO_NAMES_PATH, "r") as f:
            classes = f.read().strip().split("\n")
    except Exception as e:
        logging.error("Erro ao carregar classes YOLO: %s", e)
        sys.exit(1)
    
    # Inicializa a webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Não foi possível abrir a webcam.")
        sys.exit(1)

    # Carregar classificador de rosto do OpenCV (para outras detecções)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Inicializa a rede neural para treinamento (para aprender o que é um cell phone)
    device = get_device()
    model = CNNModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Tenta carregar um modelo treinado anteriormente
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            logging.info("Modelo carregado para continuar o treinamento!")
        except Exception as e:
            logging.error("Erro ao carregar o modelo: %s", e)

    # Transformação para pré-processar as imagens (para treinamento)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Buffers para armazenamento de imagens e rótulos (treinamento)
    image_buffer = deque(maxlen=BUFFER_SIZE)
    label_buffer = deque(maxlen=BUFFER_SIZE)
    losses = []

    def train_model_from_buffer():
        if len(image_buffer) == BUFFER_SIZE:
            images = torch.stack(list(image_buffer)).to(device)
            labels = torch.tensor(list(label_buffer), dtype=torch.float32).view(-1, 1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            logging.info(f'Perda do treinamento: {loss.item():.4f}')

    logging.info("Tudo está pronto! Iniciando o processamento...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Erro ao capturar frame da webcam.")
                break

            height, width, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # DETECÇÃO DE ROSTOS (exemplo já existente)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Rosto", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                face_region = frame[y:y + h, x:x + w]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y:y + h, x:x + w] = blurred_face
                try:
                    processed_face = transform(face_region)
                    image_buffer.append(processed_face)
                    label_buffer.append(1)
                except Exception as e:
                    logging.error("Erro ao processar imagem do rosto: %s", e)

            # DETECÇÃO DE OBJETOS COM YOLO
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=True)
            yolo_net.setInput(blob)
            try:
                detections = yolo_net.forward(output_layers)
            except Exception as e:
                logging.error("Erro na detecção YOLO: %s", e)
                continue

            # Processar objetos válidos conforme VALID_CLASSES
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        detected_class = classes[class_id]
                        if detected_class in VALID_CLASSES:
                            center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype("int")
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            label = f"{detected_class}: {confidence:.2f}"
                            
                            if detected_class == "cell phone":
                                logging.info("Cell phone detectado!")
                                # Extraindo a região do celular antes de desfocar para treinamento
                                cell_region = frame[y:y+h, x:x+w]
                                try:
                                    processed_cell = transform(cell_region)
                                    image_buffer.append(processed_cell)
                                    label_buffer.append(1)
                                except Exception as e:
                                    logging.error("Erro ao processar imagem do cell phone: %s", e)
                                # Desfocar a região do celular
                                blurred_cell = cv2.GaussianBlur(cell_region, (99, 99), 30)
                                frame[y:y+h, x:x+w] = blurred_cell
                                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            else:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Treinamento periódico
            train_model_from_buffer()
            
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        logging.info("Interrupção manual do processamento.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Salvar o modelo treinado
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logging.info("Modelo treinado salvo com sucesso!")
    except Exception as e:
        logging.error("Erro ao salvar o modelo: %s", e)

    # Gerar e salvar gráfico de aprendizado, se houver perdas registradas
    if losses:
        plt.plot(losses)
        plt.title('Gráfico de Aprendizado')
        plt.xlabel('Iterações')
        plt.ylabel('Perda')
        try:
            plt.savefig(GRAPH_SAVE_PATH)
            logging.info(f"Gráfico de aprendizado salvo em {GRAPH_SAVE_PATH}")
        except Exception as e:
            logging.error("Erro ao salvar gráfico: %s", e)
    else:
        logging.info("Nenhuma perda registrada para gerar gráfico.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplicação de Detecção com YOLO e Treinamento de Rede Neural")
    parser.add_argument("--buffer_size", type=int, default=BUFFER_SIZE, help="Tamanho do buffer para treinamento")
    args = parser.parse_args()
    BUFFER_SIZE = args.buffer_size
    main(args)
