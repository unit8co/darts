#!/usr/bin/env python3
"""
Pizza Recipe Scaler

Takes the amount of flour in grams as input and calculates
the quantities of other ingredients needed for pizza dough.
All measurements are in European units (grams, ml).

Usage:
    python pizza_recipe_scaler.py           # Interactive mode
    python pizza_recipe_scaler.py 500       # Direct mode with 500g flour
"""

import sys

# Base recipe ratios (per 100g of flour)
BASE_RATIOS = {
    "water": 0.65,          # 65% hydration (ml per g flour)
    "salt": 0.02,           # 2% salt (g per g flour)
    "yeast": 0.01,          # 1% fresh yeast or 0.3% dry yeast (g per g flour)
    "olive_oil": 0.04,      # 4% olive oil (ml per g flour)
    "sugar": 0.01,          # 1% sugar (g per g flour)
}


def calculate_ingredients(flour_grams: float) -> dict:
    """
    Calculate pizza dough ingredients based on flour amount.

    Args:
        flour_grams: Amount of flour in grams

    Returns:
        Dictionary with all ingredient quantities
    """
    ingredients = {
        "flour": flour_grams,
        "water": round(flour_grams * BASE_RATIOS["water"], 1),
        "salt": round(flour_grams * BASE_RATIOS["salt"], 1),
        "fresh_yeast": round(flour_grams * BASE_RATIOS["yeast"], 1),
        "dry_yeast": round(flour_grams * BASE_RATIOS["yeast"] * 0.33, 1),
        "olive_oil": round(flour_grams * BASE_RATIOS["olive_oil"], 1),
        "sugar": round(flour_grams * BASE_RATIOS["sugar"], 1),
    }
    return ingredients


def estimate_pizzas(flour_grams: float, pizza_size: str = "medium") -> int:
    """
    Estimate how many pizzas can be made with the given flour amount.

    Args:
        flour_grams: Amount of flour in grams
        pizza_size: Size of pizza ('small', 'medium', 'large')

    Returns:
        Estimated number of pizzas
    """
    flour_per_pizza = {
        "small": 100,   # ~25cm pizza
        "medium": 150,  # ~30cm pizza
        "large": 200,   # ~35cm pizza
    }
    return int(flour_grams // flour_per_pizza.get(pizza_size, 150))


def display_recipe(flour_grams: float) -> None:
    """
    Display the complete pizza recipe with scaled ingredients.

    Args:
        flour_grams: Amount of flour in grams
    """
    ingredients = calculate_ingredients(flour_grams)
    num_pizzas = estimate_pizzas(flour_grams)

    print("\n" + "=" * 50)
    print("       PIZZA DOUGH RECIPE")
    print("=" * 50)
    print(f"\nFor {flour_grams}g of flour, you will need:\n")
    print("-" * 30)
    print(f"  Flour:       {ingredients['flour']:.0f} g")
    print(f"  Water:       {ingredients['water']:.0f} ml (lukewarm)")
    print(f"  Salt:        {ingredients['salt']:.1f} g")
    print(f"  Olive oil:   {ingredients['olive_oil']:.0f} ml")
    print(f"  Sugar:       {ingredients['sugar']:.1f} g")
    print("-" * 30)
    print(f"  Fresh yeast: {ingredients['fresh_yeast']:.1f} g")
    print(f"    OR")
    print(f"  Dry yeast:   {ingredients['dry_yeast']:.1f} g")
    print("-" * 30)

    print(f"\nThis makes approximately {num_pizzas} medium pizza(s) (~30cm)")

    print("\n" + "=" * 50)
    print("       INSTRUCTIONS")
    print("=" * 50)
    print("""
1. Dissolve yeast and sugar in lukewarm water (35-40°C)
2. Mix flour and salt in a large bowl
3. Add the yeast mixture and olive oil
4. Knead for 10-15 minutes until smooth and elastic
5. Cover and let rise for 1-2 hours at room temperature
6. Divide into portions and shape your pizzas
7. Add toppings and bake at 250°C for 10-15 minutes
   (or as hot as your oven goes!)
""")


def main():
    """Main function to run the pizza recipe scaler."""
    print("\n🍕 PIZZA RECIPE SCALER 🍕")
    print("All measurements in European units (g, ml)\n")

    # Check for command-line argument
    if len(sys.argv) > 1:
        try:
            flour_grams = float(sys.argv[1])
            if flour_grams <= 0:
                print("Please enter a positive number.")
                sys.exit(1)
            if flour_grams < 50:
                print("Warning: Less than 50g of flour may be too little for good results.")
            display_recipe(flour_grams)
            return
        except ValueError:
            print(f"Invalid input: '{sys.argv[1]}'. Please enter a valid number.")
            sys.exit(1)

    # Interactive mode
    while True:
        try:
            user_input = input("Enter the amount of flour in grams (or 'q' to quit): ")

            if user_input.lower() in ('q', 'quit', 'exit'):
                print("\nBuon appetito! 🍕\n")
                break

            flour_grams = float(user_input)

            if flour_grams <= 0:
                print("Please enter a positive number.")
                continue

            if flour_grams < 50:
                print("Warning: Less than 50g of flour may be too little for good results.")

            display_recipe(flour_grams)

        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\nBuon appetito! 🍕\n")
            break


if __name__ == "__main__":
    main()
