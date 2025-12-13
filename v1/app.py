import pygame
import tensorflow as tf
import numpy as np
import cv2

# Configurações
LARGURA, ALTURA = 600, 450 # Ajustei para ficar mais compacto na tela
PRETO = (0,0,0)
BRANCO = (255,255,255)
CINZA = (100,100,100)
VERDE = (0, 255, 0)

pygame.init()
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Como a IA enxerga - Python AI")
fonte = pygame.font.SysFont("arial", 20)

try:
    modelo = tf.keras.models.load_model('model.h5')
    print("IA carregada com sucesso // AI load successfully")
except:
    print("Erro: Rode o script de treino primeiro! // Error: run training script first")
    exit()

# Ajustei para 280x280 para bater com a lógica de desenho abaixo
canvas_desenho = pygame.Surface((280, 280)) 
canvas_desenho.fill(PRETO)
desenhando = False
resultado_texto = "Desenhe..."

def processar_imagem(surface):
    # pegar os pixels da superficie
    img_data = pygame.surfarray.array3d(surface)
    
    # transformar em escala de cinza e rotacionar
    # CORREÇÃO: Era GBR2GRAY (errado), mudei para RGB2GRAY
    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY) 
    img_data = np.rot90(img_data)
    img_data = np.flipud(img_data)

    # redimensionar para 28x28
    img_resized = cv2.resize(img_data, (28, 28), interpolation=cv2.INTER_AREA)

    # normalizar e preparar para o keras
    img_resized = img_resized / 255.0
    img_resized = img_resized.reshape(1, 28, 28)
    return img_resized

rodando = True
while rodando:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            rodando = False
        
        # Controles do mouse
        if evento.type == pygame.MOUSEBUTTONDOWN:
            desenhando = True
        if evento.type == pygame.MOUSEBUTTONUP:
            desenhando = False
        
        # Controles do teclado
        if evento.type == pygame.KEYDOWN:
            # CORREÇÃO: Tudo isso agora está DENTRO do if KEYDOWN
            
            if evento.key == pygame.K_SPACE:
                # previsao
                imagem_ia = processar_imagem(canvas_desenho)
                predicao = modelo.predict(imagem_ia)
                numero = np.argmax(predicao)
                confianca = np.max(predicao) * 100
                resultado_texto = f"IA vê: {numero} ({confianca:.1f}%)"
                print(f"Previsão: {numero}")
            
            if evento.key == pygame.K_BACKSPACE or evento.key == pygame.K_DELETE:
                canvas_desenho.fill(PRETO)
                resultado_texto = "Limpo! Desenhe..."

    # Logica de desenho
    if desenhando:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        # Ajustar coordenada para dentro do canvas
        if 50 <= mouse_x <= 330 and 50 <= mouse_y <= 330:
            pygame.draw.circle(canvas_desenho, BRANCO, (mouse_x - 50, mouse_y - 50), 12)
    
    tela.fill(CINZA)

    # Desenha o canvas preto
    tela.blit(canvas_desenho, (50, 50))
    pygame.draw.rect(tela, BRANCO, (50, 50, 280, 280), 2) # borda

    # Textos
    texto_superficie = fonte.render(resultado_texto, True, VERDE)
    tela.blit(texto_superficie, (50, 350))

    instr_surface = fonte.render("ESPAÇO: Adivinhar | DEL: Limpar", True, BRANCO)
    tela.blit(instr_surface, (50, 20))

    pygame.display.update()

pygame.quit()