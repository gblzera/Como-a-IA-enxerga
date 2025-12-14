# Neural Network Visualizer (MNIST)

> Um visualizador interativo em tempo real que desmistifica como uma Rede Neural "enxerga" e processa dígitos manuscritos.

![Status do Projeto](https://img.shields.io/badge/Status-Finalizado-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Tech](https://img.shields.io/badge/Tech-ONNX%20%7C%20Pygame%20%7C%20OpenCV-purple)

---

## Sobre o Projeto

Este projeto foi desenvolvido com um objetivo claro: **tirar redes neurais da caixa-preta**.

Ao invés de apenas prever qual número foi desenhado, a aplicação mostra **como a IA pensa**. Cada traço no canvas passa por um pipeline completo de visão computacional e deep learning, enquanto as conexões neurais se iluminam em tempo real de acordo com a ativação dos neurônios.

Aqui não tem mágica. Tem matemática, código e visualização clara.

### Pipeline da Aplicação

1. **Input Interativo**
   Um canvas de desenho implementado em **Pygame**, onde o usuário desenha livremente um dígito.

2. **Pré-processamento Inteligente (OpenCV)**
   O desenho passa por um algoritmo robusto que:

   * Detecta automaticamente a área desenhada (*Bounding Box*)
   * Recorta apenas o número
   * Redimensiona mantendo proporção
   * Centraliza o dígito no canvas 28x28 usando centro de massa

   Resultado: a rede reconhece números mesmo tortos, deslocados ou fora de escala.

3. **Modelo Otimizado (ONNX)**
   A rede neural foi treinada em **TensorFlow/Keras** com o dataset MNIST e posteriormente convertida para **ONNX**, garantindo:

   * Inferência rápida
   * Menor consumo de recursos
   * Facilidade de deploy

4. **Visualização Neural em Tempo Real**
   As conexões entre neurônios são renderizadas dinamicamente:

   * Intensidade da cor representa o nível de ativação
   * Apenas conexões relevantes aparecem
   * Simula visualmente o caminho do "pensamento" da IA

---

## Tecnologias Utilizadas

* **Linguagem:** Python 3.12+
* **Interface Gráfica:** Pygame
* **Processamento de Imagem:** OpenCV, NumPy
* **Deep Learning:** TensorFlow (Treinamento)
* **Inferência Otimizada:** ONNX Runtime

---

## Estrutura do Projeto

```text
projeto/
│
├── models/              # Contém o cérebro da IA
│   └── cerebro.onnx        # Modelo treinado e otimizado
│
├── app_final.py            # Código principal da aplicação (Run this!)
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação
```

---

## Como Rodar Localmente

### Pré-requisitos

* Python 3 instalado (recomendado 3.10+)
* Pip atualizado

---

### Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO
```

---

### Instalar dependências

> Recomenda-se o uso de um ambiente virtual (venv).

```bash
pip install -r requirements.txt
```

---

### 4️Executar a aplicação

```bash
python app_final.py
```

---

## Como Funciona a Visualização?

A interface representa visualmente a arquitetura da rede neural:

### Camada de Entrada (Input)

* O desenho feito no grid é convertido para uma imagem **28x28**, padrão do MNIST.

### Camadas Ocultas (Hidden Layers)

* Linhas **roxas** conectam a primeira camada de neurônios
* Linhas **verdes** conectam à camada de saída
* As conexões só aparecem quando há ativação relevante

### Camada de Saída (Output)

* Representada pelas classes **0 a 9**
* A caixa destacada em verde indica a previsão final da IA

---

## Detalhes Técnicos

Para garantir robustez no reconhecimento, o pré-processamento é essencial:

```python
# Exemplo simplificado da lógica de centralização

def processar_grid(grid):
    # 1. Detecta pixels desenhados (Bounding Box)
    # 2. Recorta apenas o número
    # 3. Redimensiona mantendo aspect ratio
    # 4. Centraliza no canvas 28x28 usando centro de massa
    pass
```

Esse passo é o que separa um projetinho frágil de uma aplicação minimamente séria.

---

## requirements.txt

Crie um arquivo `requirements.txt` com o seguinte conteúdo:

```text
pygame
numpy
opencv-python
onnxruntime
tensorflow
tf2onnx
```

---

## Autor
Desenvolvido por **gblzera**
