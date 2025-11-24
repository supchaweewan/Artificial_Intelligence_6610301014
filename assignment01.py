def main():
    print("=== Pattern Recognition - Probability Theory ===")
    print("Model: Red / Blue Box + Apples & Oranges\n")

    p_red_percent = float(input("Input chance to pick from red box (%), ex: 40 : "))
    p_blue_percent = float(input("Input chance to pick from blue box (%), ex: 60 : "))

    p_red = p_red_percent / 100.0
    p_blue = p_blue_percent / 100.0

    if abs(p_red + p_blue - 1.0) > 1e-6:
        print("\n[WARN] p(red) + p(blue) != 1. Normalizing value\n")
        s = p_red + p_blue
        p_red /= s
        p_blue /= s

    print("\nInput fruits in red box")
    red_apple = int(input("  Apples in red box : "))
    red_orange = int(input("  Oranges in red box : "))

    print("\nInput fruits in blue box")
    blue_apple = int(input("  Apples in blue box : "))
    blue_orange = int(input("  Oranges in blue box : "))

    # Total
    total_red = red_apple + red_orange
    total_blue = blue_apple + blue_orange

    if total_red == 0 or total_blue == 0:
        print("\n[ERROR] One box literally has no fruits (total = 0)")
        return

    # ---------- Likelihood: p(F = a | box), p(F = o | box) ----------
    p_a_given_red = red_apple / total_red
    p_o_given_red = red_orange / total_red

    p_a_given_blue = blue_apple / total_blue
    p_o_given_blue = blue_orange / total_blue

    # ---------- Marginal: p(F = a), p(F = o) ----------
    p_F_a = p_a_given_red * p_red + p_a_given_blue * p_blue
    p_F_o = p_o_given_red * p_red + p_o_given_blue * p_blue

    # ---------- Posterior: p(red | F = o) with Bayes Method ----------
    if p_F_o == 0:
        p_red_given_o = 0.0
    else:
        p_red_given_o = (p_o_given_red * p_red) / p_F_o

    # ---------- OUTPUT ----------
    print("\n========== RESULT ==========")
    print(f"p(Box = red)   = {p_red:.4f}")
    print(f"p(Box = blue)  = {p_blue:.4f}\n")

    print(f"p(F = a | red)   = {p_a_given_red:.4f}")
    print(f"p(F = o | red)   = {p_o_given_red:.4f}")
    print(f"p(F = a | blue)  = {p_a_given_blue:.4f}")
    print(f"p(F = o | blue)  = {p_o_given_blue:.4f}\n")

    print(f"p(F = a) (Apple)  = {p_F_a:.4f}")
    print(f"p(F = o) (Orange) = {p_F_o:.4f}\n")

    print(f"p(Box = red | F = o) = {p_red_given_o:.4f}")
    print(" (Chance that we'll grab an orange from red box.)\n")

if __name__ == "__main__":
    main()