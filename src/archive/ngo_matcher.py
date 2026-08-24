import pandas as pd


# =========================================================
# LOAD NGO DATABASE
# =========================================================

ngos = pd.read_csv("data/ngos.csv")


# =========================================================
# FIND SUITABLE NGOs
# =========================================================

def find_matching_ngos(
    menu,
    quantity,
    location
):
    """
    Find NGOs that can potentially accept the surplus.

    Matching considers:
    - location
    - requested food type
    - capacity
    - pickup availability
    """

    matches = []

    for _, ngo in ngos.iterrows():

        # NGO must be in the same location for this prototype.
        if ngo["location"] != location:
            continue

        # NGO must currently have pickup capability.
        if ngo["pickup_available"] != 1:
            continue

        # Check whether the NGO accepts this food.
        accepts_food = (
            ngo["food_types"] == "Any"
            or menu in ngo["food_types"]
            or menu.split()[0] in ngo["food_types"]
        )

        if not accepts_food:
            continue

        # Do not send more than the NGO says it can accept.
        allocation = min(
            quantity,
            ngo["max_servings"]
        )

        matches.append({
            "ngo_name": ngo["ngo_name"],
            "allocation": allocation,
            "location": ngo["location"]
        })

    return matches


# =========================================================
# DISTRIBUTE SURPLUS
# =========================================================

def allocate_surplus(
    menu,
    quantity,
    location
):
    """
    Allocate surplus across matching NGOs until
    either all food is assigned or no more NGOs match.
    """

    matches = find_matching_ngos(
        menu,
        quantity,
        location
    )

    remaining = quantity
    allocations = []

    for ngo in matches:

        if remaining <= 0:
            break

        allocation = min(
            remaining,
            ngo["allocation"]
        )

        allocations.append({
            "ngo_name": ngo["ngo_name"],
            "quantity": allocation
        })

        remaining -= allocation

    return {
        "allocations": allocations,
        "unallocated_surplus": remaining
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = allocate_surplus(
        menu="Chicken Rice",
        quantity=32,
        location="Bhubaneswar"
    )

    print("\n==============================")
    print("       NGO MATCHING")
    print("==============================")

    for allocation in result["allocations"]:

        print(
            allocation["ngo_name"],
            "→",
            allocation["quantity"],
            "servings"
        )

    print(
        "\nUnallocated surplus:",
        result["unallocated_surplus"]
    )