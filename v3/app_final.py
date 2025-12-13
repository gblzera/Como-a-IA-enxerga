import pygame
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import Model

BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
CINZA_CLARO = (240, 240, 240) 
CINZA_MEDIO = (200, 200, 200) 
VERDE_NEON = (0, 200, 0)
ROXO_MAGENTA = (180, 0, 180)
AZUL_BOTAO = (70, 130, 180)

LARGURA_TELA, ALTURA_TELA = 950, 980 
GRID_VISUAL_TAM = 20
CELULA_TAM = 22 

Y_OUTPUT = 50
Y_LAYER_2 = 160
Y_LAYER_1 = 270
OFFSET_Y_GRID = 400 
OFFSET_X_GRID = (LARGURA_TELA - (GRID_VISUAL_TAM * CELULA_TAM)) // 2

pygame.init()
tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
pygame.display.set_caption("Visualização de Rede Neural - Clean Version")
fonte_num = pygame.font.SysFont("arial", 18, bold=True)
fonte_btn = pygame.font.SysFont("arial", 24, bold=True)

try:
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_modelo = os.path.join(diretorio_atual, '..', 'models', 'modelo_visual.h5')
    
    print(f"Carregando: {caminho_modelo}")
    modelo_completo = tf.keras.models.load_model(caminho_modelo)
    layer_outputs = [layer.output for layer in modelo_completo.layers if 'oculta' in layer.name or 'saida' in layer.name]
    visualizacao_model = Model(inputs=modelo_completo.input, outputs=layer_outputs)
    print("IA Pronta.")
except Exception as e:
    print(f"ERRO: Não achei o modelo em {caminho_modelo}.\nErro: {e}")
    exit()

grid_dados = np.zeros((GRID_VISUAL_TAM, GRID_VISUAL_TAM))
desenhando = False
ativacoes = None 
previsao_idx = -1
board_vazio = True
btn_rect = pygame.Rect(0, 0, 0, 0) 


def processar_grid(grid):
    if np.sum(grid) == 0:
        return np.zeros((1, 28, 28))
    
    linhas = np.any(grid, axis=1)
    colunas = np.any(grid, axis=0)
    y_min, y_max = np.where(linhas)[0][[0, -1]]
    x_min, x_max = np.where(colunas)[0][[0, -1]]

    corte = grid[y_min:y_max+1, x_min:x_max+1]
    h, w = corte.shape
    tamanho_max = 20
    
    scale = tamanho_max / max(h, w)
    novo_w, novo_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(corte, (novo_w, novo_h), interpolation=cv2.INTER_AREA)

    img_final = np.zeros((28, 28))
    pos_x = (28 - novo_w) // 2
    pos_y = (28 - novo_h) // 2
    img_final[pos_y:pos_y+novo_h, pos_x:pos_x+novo_w] = img_resized

    img_final = img_final / 255.0
    img_final = img_final.reshape(1, 28, 28) 
    
    return img_final

def calcular_posicoes_x(qtd_itens, largura_item, espaco_item):
    largura_total = qtd_itens * (largura_item + espaco_item) - espaco_item
    inicio_x = (LARGURA_TELA - largura_total) // 2
    return [inicio_x + i * (largura_item + espaco_item) for i in range(qtd_itens)]

def desenhar_tudo():
    tela.fill(BRANCO)

    vis_nodes_L1 = 30  
    vis_nodes_L2 = 20  
    
    pos_x_output = calcular_posicoes_x(11, 40, 10)
    pos_x_L1 = calcular_posicoes_x(vis_nodes_L1, 10, 8) 
    pos_x_L2 = calcular_posicoes_x(vis_nodes_L2, 10, 15) 

    if not board_vazio and ativacoes is not None:
        raw_l1 = ativacoes[0][0]
        raw_l2 = ativacoes[1][0]
        LIMIAR_VISUAL = 0.2 

        for i in range(vis_nodes_L1):
            idx_real_src = int((i / vis_nodes_L1) * len(raw_l1))
            val_src = raw_l1[idx_real_src]

            if val_src > LIMIAR_VISUAL: 
                x_start = pos_x_L1[i] + 5
                for j in range(vis_nodes_L2):
                    idx_real_dst = int((j / vis_nodes_L2) * len(raw_l2))
                    val_dst = raw_l2[idx_real_dst]
                
                    if val_dst > LIMIAR_VISUAL:
                        x_end = pos_x_L2[j] + 5
                        largura_linha = 2 if (val_src * val_dst) > 0.5 else 1
                        pygame.draw.line(tela, ROXO_MAGENTA, (x_start, Y_LAYER_1), (x_end, Y_LAYER_2), largura_linha)

        for i in range(vis_nodes_L2):
            idx_real = int((i / vis_nodes_L2) * len(raw_l2))
            val = raw_l2[idx_real]
            
            if val > LIMIAR_VISUAL:
                x_start = pos_x_L2[i] + 5
                if 0 <= previsao_idx < 10:
                    x_end = pos_x_output[previsao_idx] + 20
                    pygame.draw.line(tela, VERDE_NEON, (x_start, Y_LAYER_2), (x_end, Y_OUTPUT + 40), 2)

    for i, x in enumerate(pos_x_L1):
        cor = CINZA_CLARO
        if not board_vazio and ativacoes is not None:
             idx_real = int((i / vis_nodes_L1) * len(ativacoes[0][0]))
             val = ativacoes[0][0][idx_real]
             if val > 0.1: 
                 intensidade = min(255, int(val * 255))
                 cor = (0, intensidade, 0) 
        pygame.draw.rect(tela, cor, (x, Y_LAYER_1, 10, 10))

    for i, x in enumerate(pos_x_L2):
        cor = CINZA_CLARO
        if not board_vazio and ativacoes is not None:
             idx_real = int((i / vis_nodes_L2) * len(ativacoes[1][0]))
             val = ativacoes[1][0][idx_real]
             if val > 0.1:
                 intensidade = min(255, int(val * 255))
                 cor = (intensidade, 0, intensidade) 
        pygame.draw.rect(tela, cor, (x, Y_LAYER_2, 10, 10))

    for i in range(11):
        x = pos_x_output[i]
        bg, border, txt = BRANCO, CINZA_MEDIO, PRETO
        label = str(i) if i < 10 else "..."
        
        if i == 10 and board_vazio:
            bg, txt = PRETO, BRANCO
        elif not board_vazio and i == previsao_idx: 
            bg, txt, border = PRETO, VERDE_NEON, VERDE_NEON

        pygame.draw.rect(tela, bg, (x, Y_OUTPUT, 40, 40), border_radius=5)
        pygame.draw.rect(tela, border, (x, Y_OUTPUT, 40, 40), 2, border_radius=5)
        tela.blit(fonte_num.render(label, True, txt), (x + 12, Y_OUTPUT + 10))

    rect_grid_borda = (OFFSET_X_GRID-2, OFFSET_Y_GRID-2, GRID_VISUAL_TAM*CELULA_TAM+4, GRID_VISUAL_TAM*CELULA_TAM+4)
    pygame.draw.rect(tela, PRETO, rect_grid_borda, 2)
    
    for L in range(GRID_VISUAL_TAM):
        for C in range(GRID_VISUAL_TAM):
            cor = PRETO if grid_dados[L][C] > 0 else BRANCO
            rx = OFFSET_X_GRID + C * CELULA_TAM
            ry = OFFSET_Y_GRID + L * CELULA_TAM
            pygame.draw.rect(tela, cor, (rx, ry, CELULA_TAM, CELULA_TAM))
            pygame.draw.rect(tela, CINZA_CLARO, (rx, ry, CELULA_TAM, CELULA_TAM), 1)

    global btn_rect
    pos_y_btn = OFFSET_Y_GRID + (GRID_VISUAL_TAM * CELULA_TAM) + 30
    btn_rect = pygame.Rect(OFFSET_X_GRID, pos_y_btn, 440, 50) 
    
    pygame.draw.rect(tela, PRETO, btn_rect, border_radius=8) 
    txt_clear = fonte_btn.render("LIMPAR / CLEAR", True, BRANCO)
    tela.blit(txt_clear, (btn_rect.centerx - txt_clear.get_width()//2, btn_rect.centery - txt_clear.get_height()//2))

rodando = True
while rodando:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT: rodando = False
        
        if evento.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if btn_rect.collidepoint((mx, my)):
                grid_dados.fill(0)
                board_vazio, ativacoes = True, None
            if (OFFSET_X_GRID <= mx < OFFSET_X_GRID + GRID_VISUAL_TAM * CELULA_TAM and OFFSET_Y_GRID <= my):
                desenhando = True

        if evento.type == pygame.MOUSEBUTTONUP:
            desenhando = False
            if np.sum(grid_dados) > 0:
                board_vazio = False
                img_ia = processar_grid(grid_dados) 
                ativacoes = visualizacao_model.predict(img_ia, verbose=0)
                previsao_idx = np.argmax(ativacoes[2][0])
            else:
                board_vazio, ativacoes = True, None

    if desenhando:
        mx, my = pygame.mouse.get_pos()
        gx, gy = (mx - OFFSET_X_GRID) // CELULA_TAM, (my - OFFSET_Y_GRID) // CELULA_TAM
        if 0 <= gx < GRID_VISUAL_TAM and 0 <= gy < GRID_VISUAL_TAM:
            grid_dados[gy][gx] = 255
            if gx+1 < GRID_VISUAL_TAM: grid_dados[gy][gx+1] = 255
            if gy+1 < GRID_VISUAL_TAM: grid_dados[gy+1][gx] = 255

    desenhar_tudo()
    pygame.display.update()

pygame.quit()