import random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Núcleo Genérico do AG
# =========================
def genetic_algorithm(create_individual, fitness, selection, crossover, mutation,
                      generations=30, population_size=10):
    population = [create_individual() for _ in range(population_size)]
    history = []
    for _ in range(generations):
        new_population = []
        for _ in range(population_size):
            parent1 = selection(population, fitness)
            parent2 = selection(population, fitness)
            child = crossover(parent1, parent2)
            child = mutation(child)
            new_population.append(child)
        population = new_population
        best = max(population, key=fitness)
        history.append((best, fitness(best)))
    return history

def run_with_tracking(create_individual, fitness, selection, crossover, mutation,
                      generations=30, population_size=10):
    """Executa o AG e retorna histórico de indivíduos e fitness."""
    history = genetic_algorithm(
        create_individual, fitness, selection, crossover, mutation,
        generations=generations, population_size=population_size
    )
    # Extrai os valores de fitness ao longo das gerações
    fitness_values = [score for _, score in history]
    return history, fitness_values


def plot_fitness_evolution(fitness_values, generations, title="Evolução do Fitness"):
    """Plota a evolução do fitness ao longo das gerações e retorna figura matplotlib."""
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(range(1, generations+1), fitness_values, marker='o', color='blue')
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.grid(True)
    return fig

# =========================
# 1. Escala de Funcionários
# =========================
funcionarios = ["Ana", "Bruno", "Carlos", "Daniela", "Eduardo"]
dias = ["Seg", "Ter", "Qua", "Qui", "Sex"]

def create_individual_schedule():
    return {dia: random.choice(funcionarios) for dia in dias}

def fitness_schedule(ind):
    counts = {f: list(ind.values()).count(f) for f in funcionarios}
    diff = max(counts.values()) - min(counts.values())
    return -diff

def selection_schedule(pop, fitness):
    return max(random.sample(pop, 2), key=fitness)

def crossover_schedule(p1, p2):
    return {dia: (p1[dia] if random.random() < 0.5 else p2[dia]) for dia in dias}

def mutation_schedule(ind, rate=0.2):
    if random.random() < rate:
        ind[random.choice(dias)] = random.choice(funcionarios)
    return ind

# =========================
# 2. Cardápio Semanal
# =========================
refeicoes = ["Frango", "Peixe", "Carne", "Vegetariano", "Massa"]
dias_semana = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sab", "Dom"]
calorias = {"Frango": 500, "Peixe": 400, "Carne": 700, "Vegetariano": 350, "Massa": 600}

def create_individual_menu():
    return {dia: random.choice(refeicoes) for dia in dias_semana}

def fitness_menu(ind):
    total = sum(calorias[ref] for ref in ind.values())
    target = 3500
    return -abs(total - target)

def selection_menu(pop, fitness):
    return max(random.sample(pop, 2), key=fitness)

def crossover_menu(p1, p2):
    return {dia: (p1[dia] if random.random() < 0.5 else p2[dia]) for dia in dias_semana}

def mutation_menu(ind, rate=0.3):
    if random.random() < rate:
        ind[random.choice(dias_semana)] = random.choice(refeicoes)
    return ind

# =========================
# 3. Portfólio de Investimentos
# =========================
ativos = ["Ação A", "Ação B", "Ação C", "Ação D"]
retornos = {"Ação A": 0.1, "Ação B": 0.07, "Ação C": 0.15, "Ação D": 0.05}
risco = {"Ação A": 0.08, "Ação B": 0.05, "Ação C": 0.12, "Ação D": 0.03}

def create_individual_portfolio():
    weights = [random.random() for _ in ativos]
    total = sum(weights)
    return [w/total for w in weights]

def fitness_portfolio(ind):
    r = sum(ind[i]*retornos[a] for i, a in enumerate(ativos))
    risk = sum(ind[i]*risco[a] for i, a in enumerate(ativos))
    return r - risk

def selection_portfolio(pop, fitness):
    return max(random.sample(pop, 2), key=fitness)

def crossover_portfolio(p1, p2):
    point = random.randint(1, len(ativos)-1)
    child = p1[:point] + p2[point:]
    total = sum(child)
    return [w/total for w in child]

def mutation_portfolio(ind, rate=0.2):
    if random.random() < rate:
        i = random.randint(0, len(ind)-1)
        ind[i] += random.uniform(-0.1, 0.1)
        ind = [max(0, w) for w in ind]
        total = sum(ind)
        ind = [w/total for w in ind]
    return ind

# =========================
# Interface Streamlit
# =========================
st.title("Exemplos de Algoritmo Genético em Diferentes Contextos")

menu = st.sidebar.selectbox("Escolha a aplicação:", [
    "Escala de Funcionários",
    "Cardápio Semanal",
    "Portfólio de Investimentos",
    "Explicação das Variáveis"
])

generations = st.sidebar.slider("Número de Gerações", 10, 100, 30)
population_size = st.sidebar.slider("Tamanho da População", 5, 50, 15)

if menu == "Escala de Funcionários":
    st.header("Otimização da Escala de Funcionários")
    if st.button("Gerar Escala"):
        history, fitness_values = run_with_tracking(create_individual_schedule, fitness_schedule,
                                    selection_schedule, crossover_schedule, mutation_schedule,
                                    generations, population_size)
        best, score = history[-1]
        st.subheader(f"Melhor Escala (Fitness={score})")
        st.table(pd.DataFrame.from_dict(best, orient='index', columns=["Funcionário"]))

        st.subheader("Evolução do Fitness ao Longo das Gerações")
        fig = plot_fitness_evolution(fitness_values, generations, "Fitness - Funcionários")
        st.pyplot(fig)

elif menu == "Cardápio Semanal":
    st.header("Otimização do Cardápio Semanal")
    if st.button("Gerar Cardápio"):
        history, fitness_values = run_with_tracking(create_individual_menu, fitness_menu,
                                    selection_menu, crossover_menu, mutation_menu,
                                    generations, population_size)
        best, score = history[-1]
        st.subheader(f"Melhor Cardápio (Fitness={score})")

        df = pd.DataFrame.from_dict(best, orient='index', columns=["Refeição"])
        st.table(df)

        st.subheader("Evolução do Fitness ao Longo das Gerações")
        fig = plot_fitness_evolution(fitness_values, generations, "Fitness - Cardápio")
        st.pyplot(fig)

elif menu == "Portfólio de Investimentos":
    st.header("Design de Portfólio de Investimentos")
    if st.button("Gerar Portfólio"):
        history, fitness_values = run_with_tracking(create_individual_portfolio, fitness_portfolio,
                                    selection_portfolio, crossover_portfolio, mutation_portfolio,
                                    generations, population_size)
        best, score = history[-1]
        st.subheader(f"Melhor Portfólio (Fitness={score:.4f})")
        df = pd.DataFrame({"Ativo": ativos, "Peso": best})
        st.table(df)

        st.subheader("Evolução do Fitness ao Longo das Gerações")
        fig = plot_fitness_evolution(fitness_values, generations, "Fitness - Portfólio")
        st.pyplot(fig)

elif menu == "Explicação das Variáveis":
    st.header("O que significam as variáveis do Algoritmo Genético?")
    st.markdown("""
    ### 1. Fitness
    É uma pontuação que indica quão boa é uma solução. Cada aplicação tem seu próprio cálculo:

    - **Escala de Funcionários**: quanto mais equilibrada a distribuição, melhor.
    - **Cardápio Semanal**: quanto mais próximo das calorias desejadas, melhor.
    - **Portfólio**: maior retorno ajustado ao risco.

    ### 2. Geração
    Cada rodada de evolução. Quanto mais gerações, mais chances de melhorar.

    ### 3. População
    Quantidade de indivíduos (soluções) por geração.

    ### 4. Indivíduo
    Uma solução completa. Ex.: escala semanal, cardápio, vetor de pesos.

    ### 5. Seleção
    Escolhe os melhores indivíduos com base no fitness.

    ### 6. Crossover
    Combina dois indivíduos para formar um novo.

    ### 7. Mutação
    Introduz mudanças aleatórias para manter diversidade.
    """)
