import gymnasium as gym
import highway_env

#creaza environmentul pentru parcare
#render = True -> avem interfata grafica
#render = False -> nu avem UI(folosit doar pentru training)
# seed -> fixeaza aleatoritatea pentru a avea aceeasi stare initiala
#         (fara seed, starea initiala difera la fiecare rulare)
def create_environment(render = False, seed = None):
    env = gym.make("parking-v0",render_mode = "human" if render else None)

    #reseteaza environmentul daca seed-ul e specificat
    if seed is not None:
        env.reset(seed = seed)

    return env