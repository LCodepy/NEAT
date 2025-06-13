class Config:

    def __init__(self, name: str = "config1") -> None:
        self.name = name

        # Genome
        self.max_bias_weight_value = 10
        self.min_bias_weight_value = -10
        self.start_bias_weight_lower_bound = -1
        self.start_bias_weight_upper_bound = 1
        self.gaussian_mean = 0.0
        self.gaussian_standard_deviation = 2.0

        # Species
        self.species_improvement_threshold = 0.01
        self.species_target_size = 8
        self.species_target_step_size = 0.1

        # Mutations
        self.weight_mutation_chance = 0.8
        self.change_weight_mutation_chance = 0.9
        self.bias_mutation_chance = 0.8
        self.change_bias_mutation_chance = 0.9
        self.add_connection_mutation_chance = 0.4
        self.add_node_mutation_chance = 0.08
        self.remove_connection_mutation_chance = 0.1
        self.remove_node_mutation_chance = 0.08
        self.enable_mutation_chance = 0.25

        # Crossover
        self.excess_genes_importance = 1.0
        self.disjoint_genes_importance = 1.0
        self.weight_difference_importance = 0.4
        self.compatibility_threshold = 3
        self.survival_threshold = 0.2
        self.max_allowed_generations_since_improved = 20

    def __repr__(self) -> str:
        return f"Config('{self.name}', )"
