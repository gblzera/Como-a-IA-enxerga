import pygame
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model

LARGURA_TELA, ALTURA_TELA = 800, 750
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)
CINZA_CLARO = (200, 200, 200)
CINZA_ESCURO = (50, 50, 50)
VERDE_NEON = (57, 255, 20)
ROXO_NEON = (189, 34, 255)
AZUL_CLARO = (100, 149, 237)

GRID_VISUAL_TAM = 20 
CELULA_TAM = 18 
OFFSET_X_GRID = (LARGURA_TELA - (GRID_VISUAL_TAM * CELULA_TAM)) // 2
OFFSET_Y_GRID = 450

pygame.init()
tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
pygame.display.set_caption("Visualizador de Rede Neural")
fonte_num = pygame.font.SysFont("arial", 16, bold=True)
fonte_btn = pygame.font.SysFont("arial", 20)

try:
    modelo_completo = tf.keras.models.load_model('modelo_visual.h5')
    print("IA Carregada.")
    layer_outputs = [layer.output for layer in modelo_completo.layers if 'oculta' in layer.name or 'saida' in layer.name]
    visualizacao_model = Model(inputs=modelo_completo.input, outputs=layer_outputs)
except Exception as e:
    print(f"Erro: {e}. Rode o 'treinar_v2.py' primeiro.")
    exit()

grid_dados = np.zeros((GRID_VISUAL_TAM, GRID_VISUAL_TAM))
desenhando = False
ativacoes_atuais = None 
previsao_final = -1

btn_clear_rect = pygame.Rect(OFFSET_X_GRID, OFFSET_Y_GRID + (GRID_VISUAL_TAM * CELULA_TAM) + 20, 100, 40)

def processar_grid_para_ia(grid_20x20):
    img_temp = cv2.resize(grid_20x20, (280, 280), interpolation=cv2.INTER_NEAREST)
    img_28x28 = cv2.resize(img_temp, (28, 28), interpolation=cv2.INTER_AREA)
    img_final = img_28x28.reshape(1, 28, 28)
    return img_final

def desenhar_interface():
    tela.fill(CINZA_ESCURO)

    for L in range(GRID_VISUAL_TAM):
        for C in range(GRID_VISUAL_TAM):
            cor = BRANCO if grid_dados[L][C] > 0 else PRETO
            rect_X = OFFSET_X_GRID + C * CELULA_TAM
            rect_Y = OFFSET_Y_GRID + L * CELULA_TAM
            pygame.draw.rect(tela, cor, (rect_X, rect_Y, CELULA_TAM-1, CELULA_TAM-1))
            pygame.draw.rect(tela, CINZA_CLARO, (rect_X, rect_Y, CELULA_TAM, CELULA_TAM), 1) # Borda

    pygame.draw.rect(tela, AZUL_CLARO, btn_clear_rect, border_radius=5)
    texto_btn = fonte_btn.render("Clear", True, BRANCO)
    tela.blit(texto_btn, (btn_clear_rect.centerx - texto_btn.get_width()//2, btn_clear_rect.centery - texto_btn.get_height()//2))

    visualizar_rede()

def visualizar_rede():
    TOPO_Y = 50
    LAYER_1_Y = 180
    LAYER_2_Y = 280
    
    largura_box = 40
    espaco = 10
    inicio_x_saida = (LARGURA_TELA - (10 * (largura_box + espaco))) // 2

    for i in range(10):
        pos_x = inicio_x_saida + i * (largura_box + espaco)
        
        cor_box = BRANCO
        if ativacoes_atuais and previsao_final == i:
            certeza = ativacoes_atuais[2][0][i]
            cor_box = (int(255 * (1-certeza)), 255, int(255 * (1-certeza))) # Fica verde

        pygame.draw.rect(tela, cor_box, (pos_x, TOPO_Y, largura_box, largura_box), border_radius=4)
        pygame.draw.rect(tela, CINZA_CLARO, (pos_x, TOPO_Y, largura_box, largura_box), 2, border_radius=4)
        texto = fonte_num.render(str(i), True, PRETO if cor_box != PRETO else BRANCO)
        tela.blit(texto, (pos_x + 15, TOPO_Y - 20))

    if ativacoes_atuais is None: return

    out_h1 = ativacoes_atuais[0][0] 
    out_h2 = ativacoes_atuais[1][0] 

    def desenhar_camada_pontos(ativacoes, y_pos, qtd_para_mostrar, cor_tema):
        largura_ponto = 8
        espaco_ponto = 4
        inicio_x = (LARGURA_TELA - (qtd_para_mostrar * (largura_ponto + espaco_ponto))) // 2
        
        ativ_norm = ativacoes / np.max(ativacoes) if np.max(ativacoes) > 0 else ativacoes

        for i in range(qtd_para_mostrar):
            if i >= len(ativ_norm): break
            
            intensidade = int(ativ_norm[i] * 255)
            cor_ativa = (intensidade, intensidade, intensidade)
            if intensidade > 50: 
                 cor_ativa = (cor_tema[0], int(cor_tema[1] * (intensidade/255)), cor_tema[2])

            pos_x = inicio_x + i * (largura_ponto + espaco_ponto)
            pygame.draw.rect(tela, cor_ativa, (pos_x, y_pos, largura_ponto, largura_ponto))
            
            if y_pos == LAYER_2_Y:
                 destino_x = inicio_x_saida + (i % 10) * (largura_box + espaco) + largura_box//2
                 pygame.draw.line(tela, (50,50,50), (pos_x+largura_ponto//2, y_pos), (destino_x, TOPO_Y+largura_box), 1)

    desenhar_camada_pontos(out_h2, LAYER_2_Y, 64, ROXO_NEON)
    desenhar_camada_pontos(out_h1, LAYER_1_Y, 80, VERDE_NEON) 

rodando = True
clock = pygame.time.Clock()

while rodando:
    clock.tick(60)
    
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            rodando = False
        
        if evento.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            if btn_clear_rect.collidepoint(mouse_pos):
                grid_dados.fill(0)
                ativacoes_atuais = None
                previsao_final = -1
            
            if (OFFSET_X_GRID <= mouse_pos[0] < OFFSET_X_GRID + GRID_VISUAL_TAM * CELULA_TAM and
                OFFSET_Y_GRID <= mouse_pos[1] < OFFSET_Y_GRID + GRID_VISUAL_TAM * CELULA_TAM):
                desenhando = True

        if evento.type == pygame.MOUSEBUTTONUP:
            desenhando = False
            if np.sum(grid_dados) > 0: 
                img_ia = processar_grid_para_ia(grid_dados)
                ativacoes_atuais = visualizacao_model.predict(img_ia, verbose=0)
                saida_final = ativacoes_atuais[2][0]
                previsao_final = np.argmax(saida_final)

    if desenhando:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        grid_x = (mouse_x - OFFSET_X_GRID) // CELULA_TAM
        grid_y = (mouse_y - OFFSET_Y_GRID) // CELULA_TAM
        
        if 0 <= grid_x < GRID_VISUAL_TAM and 0 <= grid_y < GRID_VISUAL_TAM:
            grid_dados[grid_y][grid_x] = 255
            if grid_x + 1 < GRID_VISUAL_TAM: grid_dados[grid_y][grid_x+1] = 150
            if grid_y + 1 < GRID_VISUAL_TAM: grid_dados[grid_y+1][grid_x] = 150
            if grid_x > 0: grid_dados[grid_y][grid_x-1] = 150
            if grid_y > 0: grid_dados[grid_y-1][grid_x] = 150

    desenhar_interface()
    pygame.display.update()

pygame.quit()