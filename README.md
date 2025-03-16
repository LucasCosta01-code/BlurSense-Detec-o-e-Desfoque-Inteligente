# BlurSense: Detecção e Desfoque Inteligente

## Sobre o Projeto

O BlurSense é um sistema de detecção de objetos em tempo real que utiliza YOLO (You Only Look Once) e redes neurais convolucionais (CNN) para identificar e proteger informações sensíveis em vídeo. O sistema detecta rostos, celulares e outros objetos específicos, aplicando automaticamente um desfoque gaussiano para preservar a privacidade.

## Funcionalidades

- **Detecção de objetos em tempo real** utilizando YOLO
- **Reconhecimento facial** com OpenCV
- **Desfoque automático** de rostos e dispositivos eletrônicos
- **Treinamento contínuo** de uma CNN para melhorar a detecção
- **Visualização de métricas** de aprendizado

## Requisitos

O sistema verifica e instala automaticamente os seguintes pacotes:
- opencv-python
- torch
- torchvision
- matplotlib
- numpy

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/BlurSense.git
cd BlurSense
```

2. Execute o script principal:
```bash
python main.py
```

O programa baixará automaticamente os arquivos necessários do YOLO na primeira execução.

## Uso

Ao iniciar, o programa:
1. Ativa a webcam para captura de vídeo em tempo real
2. Detecta objetos e rostos no vídeo
3. Aplica desfoque em rostos e celulares detectados
4. Treina continuamente o modelo para melhorar a detecção
5. Salva o modelo treinado e gráficos de aprendizado

Para encerrar o programa, pressione 'q' na janela de visualização.

## Parâmetros

O programa aceita os seguintes parâmetros:
- `--buffer_size`: Define o tamanho do buffer para treinamento (padrão: 100)

Exemplo:
```bash
python main.py --buffer_size 200
```

## Estrutura de Arquivos

- `main.py`: Script principal
- `yolov3-tiny.cfg`: Arquivo de configuração do YOLO
- `yolov3-tiny.weights`: Pesos pré-treinados do YOLO
- `coco.names`: Classes de objetos do COCO dataset
- `treinamento/`: Diretório para armazenar modelos treinados e gráficos

## Como Funciona

1. **Detecção de Objetos**: Utiliza YOLO para identificar objetos em tempo real
2. **Reconhecimento Facial**: Usa o classificador Haar Cascade do OpenCV
3. **Proteção de Privacidade**: Aplica desfoque gaussiano em áreas sensíveis
4. **Aprendizado Contínuo**: Treina uma CNN para melhorar a detecção de objetos específicos

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

## Autor

Desenvolvido por  MSCODEX
