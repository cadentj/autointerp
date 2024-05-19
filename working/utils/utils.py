def format_list(
    l: list,
    leading: str = "",
) -> str:
    formatted_str = ""
    for i, item in enumerate(l):
        formatted_str += f"{leading} {i}: {item}\n"
    return formatted_str