
def elitist_selection(population, num_elitist = 0):
    num_population = len(population)
    if num_elitist == 0:
        num_elitist = int(num_population/2)
    population = sorted(population, key=lambda obj: obj.fitness)
    return population