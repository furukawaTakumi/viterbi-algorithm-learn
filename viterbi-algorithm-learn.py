
# https://spjai.com/viterbi-algorithm/
# 上記URLのコードを理解しながらビタビアルゴリズムを理解するためにコードを読みながらコメントを追加していった．

states = ('Rainy', 'Sunny', 'Cloudy')

observations = ('sleep', 'game', 'eat')

start_probability = {'Rainy': 0.3, 'Sunny': 0.4, 'Cloudy': 0.3}


transition_probability = {
    'Rainy' : {'Rainy': 0.4, 'Sunny': 0.3, 'Cloudy': 0.3},
    'Sunny' : {'Rainy': 0.2, 'Sunny': 0.7, 'Cloudy': 0.1},
    'Cloudy' : {'Rainy': 0.4, 'Sunny': 0.1, 'Cloudy': 0.5}
}

emission_probability = {
    'Rainy' : {'sleep': 0.4, 'game': 0.4, 'eat': 0.2},
    'Sunny' : {'sleep': 0.2, 'game': 0.7, 'eat': 0.1},
    'Cloudy' : {'sleep': 0.2, 'game': 0.2, 'eat': 0.6},
}


def forward_viterbi(y, X, sp, tp, ep):
    T = {}
    for state in X:
        T[state] = (sp[state], [state], sp[state]) # 各状態について最も発生しやすい経路とその確率を保持する変数Tを初期化している
    for output in y:
        U = {}
        for next_state in X:
            total = 0
            argmax = None
            valmax = 0
            for source_state in X:
                (prob, v_path, v_prob) = T[source_state]
                p = ep[source_state][output] * tp[source_state][next_state] # 観測値から潜在状態への確率と，遷移確率をかけることで，その観測値が得られる確率を計算
                prob *= p # 一つ前の観測値であるTになる確率と，今計算した経路の確率をかけることで，その経路になる確率を計算
                v_prob *= p # 上と同様
                total += prob
                if v_prob > valmax: # 最大の確率を保持するパスと確率を保持する
                    argmax = v_path + [next_state]
                    valmax = v_prob
            U[next_state] = (total, argmax, valmax)
        T = U # 計算結果Uを用いて次のUを計算したいので，こうする．
    ## apply sum/max to the final states:
    total = 0
    argmax = None
    valmax = 0
    for state in X: # 各状態についての最大の確率をもつstateを探している
        (prob, v_path, v_prob) = T[state]
        total += prob
        if v_prob > valmax:
            argmax = v_path
            valmax = v_prob
    return (total, argmax, valmax)






result = forward_viterbi(('sleep', 'sleep', 'eat'), states, start_probability, transition_probability, emission_probability)


print(result)