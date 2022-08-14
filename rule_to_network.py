
class Predicate:
    def __init__(self, name, negated=False):
        self.name = name
        self.negated = negated

class Rule:
    def __init__(self, head, body):
        self.head = head
        self.body = body

def load_rules(filename):
    def cleanse(str):
        """Sanitize a string rule and remove stopwords
        """
        rep = ['\n', '-', ' ', '.']
        for r in rep:
            str = str.replace(r, '')
        return str

    file = open(filename, 'rt', encoding='UTF8')
    ruleset = []
    for line in file:
        tokens = line.split(':')
        head = Predicate(cleanse(tokens[0]))
        body = []
        for obj in tokens[1].split(','):
            obj = cleanse(obj)
            negated = False
            if obj.startswith('not'):
                negated = True
                obj = obj.replace('not', '')
            predicate = Predicate(cleanse(obj), negated=negated)
            body.append(predicate)
        rule = Rule(head, body)
        ruleset.append(rule)
    file.close()
    return ruleset

def rewrite_rules(ruleset):
    dict = {}
    for rule in ruleset:
        if rule.head.name not in dict:
            dict[rule.head.name] = 1
        else:
            dict[rule.head.name] += 1

    rewritten_rules = []
    i = len(ruleset)
    for rule in ruleset[:]:
        if dict[rule.head.name] > 1:
            new_predicate = Predicate(rule.head.name + str(i))
            rewritten_rules.append(Rule(rule.head, [new_predicate]))
            rewritten_rules.append(Rule(new_predicate, rule.body))
            ruleset.remove(rule)
            i += 1
    del dict

    return ruleset + rewritten_rules

def get_antecedents(rules):
    all_antecedents = []
    for rule in rules:
        for predicate in rule.body:
            if predicate.name not in all_antecedents:
                all_antecedents.append(predicate.name)
    return all_antecedents

def get_consequents(rules):


    all_consequents = []
    for rule in rules:
        if rule.head.name not in all_consequents:
            all_consequents.append(rule.head.name)
    return all_consequents

def rule_to_network(ruleset):

    rule_layers = []
    l = 0
    copied_rules = ruleset.copy()
    while len(copied_rules) > 0:
        if l == 0:
            all_antecedents = get_antecedents(copied_rules)
        else:
            all_antecedents = get_antecedents(rule_layers[-1])

        rule_layer = []
        for rule in copied_rules[:]:
            if rule.head.name not in all_antecedents:
                rule_layer.append(rule)
                copied_rules.remove(rule)
        del all_antecedents[:]
        rule_layers.append(rule_layer)

    rule_layers = rule_layers[::-1]

    omega = DEFAULT_WEIGHT
    weights = []
    biases = []
    layers = []
    last_layer = []

    for rule_layer in rule_layers:

        current_layer = get_antecedents(rule_layer)
        next_layer = get_consequents(rule_layer)

        for unit in current_layer:
            if unit not in last_layer:
                last_layer.append(unit)
        current_layer = last_layer.copy()

        layers.extend([current_layer, next_layer])
        last_layer = next_layer.copy()

        # Store the occurrence of consequences. For example,
        # if a consequent occurred more than one, then it is a disjunctive rule
        dict = {}
        for rule in rule_layer:
            if rule.head.name not in dict:
                dict[rule.head.name] = 1
            else:
                dict[rule.head.name] += 1

        weight = np.zeros([len(current_layer), len(next_layer)])
        bias = np.zeros(len(next_layer))

        for rule in rule_layer:

            j = next_layer.index(rule.head.name)
            for predicate in rule.body:
                i = current_layer.index(predicate.name)
                if predicate.negated:
                    weight[i][j] = -omega
                else:
                    weight[i][j] = omega

            if dict[rule.head.name] > 1:
                bias[j] = 0.5 * omega
            else:
                p = len(rule.body)
                bias[j] = (p - 0.5) * omega

        weights.append(np.array(weight))
        biases.append(np.array([bias]))

    return weights, biases, layers
