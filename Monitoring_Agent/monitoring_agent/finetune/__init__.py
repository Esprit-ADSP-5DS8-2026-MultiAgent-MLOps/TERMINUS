"""
Fine-tuning du diagnostiqueur local (Qwen3-4B-Instruct-2507, LoRA avec MLX sur Mac Apple Silicon).

    bases.py      jeux de données « sains » + un modèle substitut entraîné dessus (le modèle « déployé »)
    scenarios.py  on casse la production de façon contrôlée (shift, concept, qualité, labels en retard)
                  puis le vrai moteur d'analyse calcule les signaux ; l'oracle (= la procédure) étiquette
    teacher.py    un grand modèle (Groq) rédige le diagnostic ; on ne garde que ce qui est juste et vérifié
    build.py      découpage par jeu de données (aucune fuite), équilibrage, fichiers MLX, config LoRA

Le jeu de test (car_loan, car_price) n'est jamais vu à l'entraînement : il sert à l'évaluation à 4 voies
(règles / SLM de base / SLM fine-tuné / grand LLM).
"""
