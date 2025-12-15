import gymnasium as gym
import highway_env
import time

#se creeaza environmentul de parcare
#parking-v0 este un environment special gata facut
#render_mode = "human" este pentru a afisa interfata grafica
env = gym.make("parking-v0",render_mode = "human")

#pornesc un episod nou
#obs = stare initiala(ce vede agentul cand se deschide harta)
#info = informatii extra
obs,info = env.reset()

#rulaz 300 de pasi in environment
for _ in range(300):
    #alagem o actiune random din spatiul de actiuni(provizoriu)
    action = env.action_space.sample()

    #aplic actiunea in environment(adica, environmentul muta masina, calculeza rewardul, decide daca episodul s-a terminat)
    obs,reward, terminated, truncated, info = env.step(action)

    #daca episodul s-a terminat(parcare reusita, coliziune, timeout) -> pornesc un episod nou
    if terminated or truncated:
        obs, info = env.reset()

    time.sleep(0.05)

env.close()
