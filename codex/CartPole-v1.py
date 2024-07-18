import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gymnasium as gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from IPython import display

# Deep Q-Learning parametreleri
gamma = 0.95  # İndirim oranı
epsilon = 1.0  # Keşif oranı başlangıç değeri
epsilon_min = 0.01  # Minimum keşif oranı
epsilon_decay = 0.995  # Keşif oranı azalma hızı
learning_rate = 0.001  # Öğrenme hızı
batch_size = 64  # Mini-batch boyutu
memory_size = 2000  # Replay buffer boyutu

# Ortamı oluştur
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]  # Durum uzayının boyutu
action_size = env.action_space.n  # Aksiyon uzayının boyutu

# Replay buffer (deneyim tekrarı için)
memory = deque(maxlen=memory_size)  # FIFO veri yapısı

# Q-ağını (sinir ağı) oluştur
model = Sequential()
model.add(Input(shape=(state_size,)))
model.add(Dense(24, activation='relu'))  # İlk gizli katman
model.add(Dense(24, activation='relu'))  # İkinci gizli katman
model.add(Dense(action_size, activation='linear'))  # Çıkış katmanı
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))  # Modeli derle

# Ajanın hareketini belirleyen politika fonksiyonu
def act(state):
    if np.random.rand() <= epsilon:  # Epsilon-greedy politika
        return env.action_space.sample()  # Rastgele aksiyon seç
    act_values = model.predict(state)  # Modelden aksiyon değerlerini tahmin et
    return np.argmax(act_values[0])  # En iyi aksiyonu seç

# Deneyimleri replay buffer'a ekleme ve ağın eğitimi
def replay():
    global epsilon
    if len(memory) < batch_size:  # Replay buffer yeterince büyük değilse geri dön
        return
    minibatch = random.sample(memory, batch_size)  # Replay buffer'dan rastgele örnekler al
    for state, action, reward, next_state, done in minibatch:
        target = reward  # Hedef ödül
        if not done:
            target = (reward + gamma * np.amax(model.predict(next_state)[0]))  # Hedefi güncelle
        target_f = model.predict(state)  # Mevcut tahmin
        target_f[0][action] = target  # Tahmini hedef ile güncelle
        model.fit(state, target_f, epochs=1, verbose=0)  # Modeli eğit
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay  # Epsilon'u azalt

# Görselleştirme fonksiyonu
def render(env):
    plt.imshow(env.render(mode='rgb_array'))  # Ortamın görüntüsünü al
    display.display(plt.gcf())  # Görüntüyü göster
    display.clear_output(wait=True)  # Önceki görüntüyü temizle

# Ajanı çalıştır
episodes = 1000  # Eğitim döngüsü sayısı
scores = []  # Bölüm sonuçlarını depolamak için liste
for e in range(episodes):
    state = env.reset()  # Ortamı sıfırla ve başlangıç durumunu al
    state = np.reshape(state, [1, state_size])  # Durumu şekillendir
    score = 0  # Bölüm puanı
    for time in range(500):  # Her bölümde maksimum adım sayısı
        action = act(state)  # Politika fonksiyonu ile aksiyonu belirle
        next_state, reward, done, _ = env.step(action)  # Aksiyonu uygula ve yeni durumu al
        reward = reward if not done else -10  # Ödülü güncelle
        next_state = np.reshape(next_state, [1, state_size])  # Yeni durumu şekillendir
        memory.append((state, action, reward, next_state, done))  # Deneyimi replay buffer'a ekle
        state = next_state  # Durumu güncelle
        score += reward  # Puanı güncelle
        if done:  # Eğer bölüm sonlandıysa
            print(f"episode: {e}/{episodes}, score: {score}, e: {epsilon:.2}")  # Bölüm bilgisini yazdır
            scores.append(score)  # Bölüm sonucunu kaydet
            break
        replay()  # Modeli eğit

    # Her 50 bölümde bir performansı göster
    if e % 50 == 0:
        render(env)  # Ortamı görselleştir

env.close()  # Ortamı kapat

# Eğitim sonuçlarını çizdir
plt.figure(figsize=(10, 6))
plt.plot(scores, marker='o', linestyle='-', color='b')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Deep Q-Learning - CartPole-v1')
plt.grid(True)
plt.show()
