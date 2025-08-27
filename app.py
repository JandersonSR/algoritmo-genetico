import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Exemplos de Algoritmo Genético em Diferentes Aplicações")
st.write("Escolha uma aplicação no menu lateral para ver como Algoritmo Genético pode ser utilizado em contextos distintos.")

# --- Menu lateral ---
app_choice = st.sidebar.selectbox(
    "Escolha a aplicação",
    ["Escala de Funcionários", "Otimização de Cardápio", "Portfólio de Investimentos", "Como funciona o Algoritmo Genético"]
)

# ============================
# 1️⃣ Escala de Funcionários
# ============================
if app_choice == "Escala de Funcionários":
    st.header("Escala de Funcionários (Turnos Coloridos)")
    employees = ["A", "B", "C", "D", "E"]
    days = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sab", "Dom"]
    shifts = ["Manhã", "Tarde", "Noite"]

    def create_individual(): return {day: {shift: random.choice(employees) for shift in shifts} for day in days}
    def fitness(schedule):
        score = 0
        for e in employees:
            work_shifts = sum([1 for day in days for shift in shifts if schedule[day][shift]==e])
            score += 1 if work_shifts<=3 else -(work_shifts-3)
        if "A" in schedule["Dom"].values(): score -= 2
        return score
    def selection(pop, k=3): return max(random.sample(pop, k), key=fitness)
    def crossover(p1, p2):
        child = {}
        for i, day in enumerate(days):
            child[day] = p1[day] if i < len(days)//2 else p2[day]
        return child
    def mutation(ind, rate=0.2):
        if random.random() < rate:
            day, shift = random.choice(days), random.choice(shifts)
            ind[day][shift] = random.choice(employees)
        return ind
    def genetic_algorithm(generations=30, population_size=10):
        pop = [create_individual() for _ in range(population_size)]
        history = []
        for _ in range(generations):
            new_pop = [mutation(crossover(selection(pop), selection(pop))) for _ in range(population_size)]
            pop = new_pop
            best = max(pop, key=fitness)
            history.append((best, fitness(best)))
        return history

    generations = st.slider("Número de gerações", 10, 100, 30)
    population_size = st.slider("Tamanho da população", 5, 20, 10)

    if st.button("Gerar escala"):
        history = genetic_algorithm(generations=generations, population_size=population_size)
        best_schedule, best_score = history[-1]
        st.subheader(f"Melhor escala encontrada (Fitness = {best_score})")

        rows = [[day, shift, best_schedule[day][shift]] for day in days for shift in shifts]
        df = pd.DataFrame(rows, columns=["Dia", "Turno", "Funcionário"])
        df_pivot = df.pivot(index="Turno", columns="Dia", values="Funcionário")
        color_map = {"A": "#FF9999", "B": "#99CCFF", "C": "#99FF99", "D": "#FFD699", "E": "#D699FF"}
        st.dataframe(df_pivot.style.applymap(lambda val: f"background-color: {color_map.get(val,'#FFFFFF')}"))

# ============================
# 2️⃣ Otimização de Cardápio
# ============================
elif app_choice == "Otimização de Cardápio":
    st.header("Otimização de Cardápio Semanal")
    meals = ["Salada", "Frango", "Peixe", "Carne", "Arroz", "Feijão", "Legumes"]
    days = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sab", "Dom"]
    meal_calories = {"Salada":100, "Frango":250, "Peixe":200, "Carne":300, "Arroz":180, "Feijão":150, "Legumes":120}

    def create_individual(): return {day: random.choice(meals) for day in days}
    def fitness(schedule): return -abs(1500 - sum([meal_calories[m] for m in schedule.values()]))
    def selection(pop, k=3): return max(random.sample(pop, k), key=fitness)
    def crossover(p1, p2): return {day: p1[day] if i<len(days)//2 else p2[day] for i, day in enumerate(days)}
    def mutation(ind, rate=0.3):
        if random.random()<rate:
            day=random.choice(days)
            ind[day]=random.choice(meals)
        return ind
    def genetic_algorithm(generations=30, population_size=10):
        pop=[create_individual() for _ in range(population_size)]
        history=[]
        for _ in range(generations):
            new_pop=[mutation(crossover(selection(pop), selection(pop))) for _ in range(population_size)]
            pop=new_pop
            best=max(pop,key=fitness)
            history.append((best, fitness(best)))
        return history

    generations = st.slider("Número de gerações", 10, 100, 30, key="menu_cardapio")
    population_size = st.slider("Tamanho da população", 5, 20, 10, key="menu_cardapio2")

    if st.button("Gerar Cardápio"):
        history = genetic_algorithm(generations=generations, population_size=population_size)
        best_schedule, best_score = history[-1]
        st.subheader(f"Melhor cardápio semanal (Fitness = {best_score})")
        df = pd.DataFrame.from_dict(best_schedule, orient='index', columns=["Refeição"])
        st.table(df)

# ============================
# 3️⃣ Portfólio de Investimentos
# ============================
elif app_choice == "Portfólio de Investimentos":
    st.header("Design de Portfólio de Investimentos")
    assets = ["Ação1","Ação2","Ação3","Ação4","Ação5"]
    returns=[0.1,0.08,0.12,0.07,0.09]
    risks=[0.05,0.03,0.06,0.02,0.04]

    def create_individual(): return np.random.dirichlet(np.ones(len(assets)))
    def fitness(weights): return sum([w*r for w,r in zip(weights,returns)])-sum([w*s for w,s in zip(weights,risks)])
    def selection(pop, k=3): return max(random.sample(pop,k), key=fitness)
    def crossover(p1,p2): alpha=random.random(); return alpha*p1+(1-alpha)*p2
    def mutation(ind, rate=0.2):
        if random.random()<rate:
            idx=random.randint(0,len(ind)-1)
            ind[idx]=random.random()
            ind/=sum(ind)
        return ind
    def genetic_algorithm(generations=30, population_size=10):
        pop=[create_individual() for _ in range(population_size)]
        history=[]
        for _ in range(generations):
            new_pop=[mutation(crossover(selection(pop),selection(pop))) for _ in range(population_size)]
            pop=new_pop
            best=max(pop,key=fitness)
            history.append((best, fitness(best)))
        return history

    generations = st.slider("Número de gerações", 10, 100, 30, key="menu_portfolio")
    population_size = st.slider("Tamanho da população", 5, 20, 10, key="menu_portfolio2")

    if st.button("Gerar Portfólio"):
        history = genetic_algorithm(generations=generations, population_size=population_size)
        best_weights, best_score = history[-1]
        st.subheader(f"Melhor portfólio (Fitness = {best_score:.4f})")
        df = pd.DataFrame({"Ativo": assets, "Peso": [f"{w:.2f}" for w in best_weights]})
        st.table(df)
# ============================
# 4️⃣ Como funciona o Algoritmo Genético (com explicação detalhada)
# ============================
elif app_choice == "Como funciona o Algoritmo Genético":
    st.header("Como funciona o Algoritmo Genético")

    st.markdown("""
    O **Algoritmo Genético (AG)** é inspirado na evolução natural. Ele busca soluções ótimas através de ciclos de seleção, reprodução e mutação.
    """)

    st.subheader("Fluxo do Algoritmo")

    # --- Diagrama com matplotlib ---
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlim(0,10)
    ax.set_ylim(0,6)
    ax.axis('off')

    def draw_box(x, y, text, color="#87CEEB"):
        rect = mpatches.FancyBboxPatch((x,y),2,1,boxstyle="round,pad=0.1", facecolor=color, edgecolor="black")
        ax.add_patch(rect)
        ax.text(x+1, y+0.5, text, ha='center', va='center', fontsize=10, weight='bold')

    draw_box(4,5, "População Inicial")
    draw_box(4,3.8, "Calcular Fitness")
    draw_box(4,2.6, "Seleção")
    draw_box(4,1.4, "Crossover")
    draw_box(4,0.2, "Mutação")
    draw_box(4,-1, "Nova População")

    # Conectar os passos
    ax.annotate("", xy=(5,4.8), xytext=(5,5), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5,3.6), xytext=(5,3.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5,2.4), xytext=(5,2.6), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5,1.2), xytext=(5,1.4), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5,0), xytext=(5,0.2), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5,-0.8), xytext=(5,-1), arrowprops=dict(arrowstyle="->"))

    st.pyplot(fig)

    st.subheader("Explicação das Variáveis e Conceitos")

    st.markdown("""
**Fitness**
O fitness é uma pontuação que indica quão boa é uma solução (indivíduo).
Cada aplicação tem seu próprio critério de fitness:

| Aplicação | Como é calculado | Objetivo |
|-----------|-----------------|----------|
| Escala de funcionários | Penaliza excesso de turnos por funcionário e violações de preferência | Quanto maior, melhor; mostra que a escala atende às regras |
| Cardápio semanal | Penaliza a diferença entre calorias da semana e valor desejado | Quanto maior (menos penalidade), melhor equilibrado está o cardápio |
| Portfólio de investimentos | Calcula retorno - risco do portfólio | Quanto maior, melhor o retorno ajustado ao risco |

Resumo: Fitness é uma medida de qualidade da solução. O algoritmo sempre tenta **maximizar o fitness**.

---

**Outras variáveis importantes**

**a) Geração (`generation`)**
Representa uma rodada de evolução da população. Cada geração cria uma nova população a partir da anterior. Serve para controlar quantas vezes o algoritmo tenta melhorar as soluções.

**b) População (`population_size`)**
Número de indivíduos em cada geração.
- Mais indivíduos → mais diversidade → maior chance de encontrar boas soluções.
- Menos indivíduos → execução mais rápida, mas risco de soluções ruins.

**c) Indivíduo**
Representa uma solução completa do problema.
Ex.:
- Escala de funcionários → um conjunto de turnos para todos os dias.
- Cardápio → uma escolha de refeições para cada dia da semana.
- Portfólio → um vetor com os pesos de cada ativo.

**d) Seleção (`selection`)**
Processo de escolher os indivíduos “mais aptos” para gerar filhos. Normalmente baseado no fitness. Quem tem fitness maior tem mais chances de ser selecionado.

**e) Crossover (Recombinação)**
Mistura dois indivíduos para criar um filho.
Ex.: metade da semana de um pai, metade do outro (escala ou cardápio). Permite combinar características boas de diferentes soluções.

**f) Mutação (`mutation`)**
Alteração aleatória em um indivíduo.
Evita que todas as soluções fiquem iguais e aumenta a diversidade genética.
Ex.: trocar um funcionário em um turno, mudar uma refeição ou alterar um peso de ativo.
""")
