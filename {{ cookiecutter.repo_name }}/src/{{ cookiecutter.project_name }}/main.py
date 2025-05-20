import argparse

def main(args) -> None:
    """
    Main function that prints a greeting message.

    Args:
        name (str): The name to greet. Defaults to "World".
    """
    
    print(f"Hello, {args.name}! You are {args.age} years old.")
    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main entry point for the project.")
    parser.add_argument("--name", '-n', type=str, default="World", help="Name to greet")
    parser.add_argument("--age", '-a', type=int, default=0, help="Age of the person")
    args = parser.parse_args()

    main(args)