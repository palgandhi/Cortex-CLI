import sys
import argparse
import os
import pyfiglet
from colorama import init, Fore, Style
from cortex.data_handlers.detector import detect_dataset_type
from cortex.nlp.parser import parse_user_intent
from cortex.algorithms.registry import get_suggested_models
from cortex.pipeline.main import run_training_pipeline

def main():
    init(autoreset=True)

    parser = argparse.ArgumentParser(
        description="Cortex CLI: A powerful tool for machine learning from the command line."
    )
    
    parser.add_argument(
        "dataset",
        nargs='?',
        type=str,
        help="Path to the dataset (e.g., data.csv or /path/to/image_folder)."
    )
    
    parser.add_argument(
        "--auto-run",
        action="store_true",
        help="Run the pipeline non-interactively using default suggestions."
    )
    
    args = parser.parse_args()
    
    print(Fore.BLUE + Style.BRIGHT + "-" * 70)
    banner = pyfiglet.figlet_format("Cortex", font="standard")
    print(Fore.CYAN + Style.BRIGHT + banner)
    print(Fore.BLUE + Style.BRIGHT + " " * 15 + "The Command-Line ML Engine")
    print("-" * 70 + Style.RESET_ALL)
    print(f"\n{Fore.GREEN}Ready to automate your machine learning tasks.")
    print(f"{Fore.YELLOW}Type 'exit' or 'quit' at any prompt to terminate.\n")
    
    while True:
        dataset_path = args.dataset
        
        if not dataset_path:
            dataset_path = input(f"{Fore.CYAN}Please enter the path to your dataset: {Style.RESET_ALL}").strip()
            if dataset_path.lower() in ('exit', 'quit'):
                print(f"\n{Fore.RED}Goodbye! 👋{Style.RESET_ALL}")
                break
        
        if not os.path.exists(dataset_path):
            print(f"{Fore.RED}Error: The path '{dataset_path}' does not exist. Please try again.{Style.RESET_ALL}")
            args.dataset = None
            continue

        print(f"\n{Fore.MAGENTA}" + "="*50)
        print("  DATASET HANDLING")
        print("="*50 + Style.RESET_ALL)
        
        handler = detect_dataset_type(dataset_path)
        
        if handler:
            print(f"{Fore.GREEN}Dataset type detected: {handler.detect_type()}{Style.RESET_ALL}")
            
            user_input = input(f"\n{Fore.CYAN}What do you want to do with this dataset? (e.g., 'I want to predict house prices'): {Style.RESET_ALL}").strip()
            if user_input.lower() in ('exit', 'quit'):
                print(f"\n{Fore.RED}Goodbye! 👋{Style.RESET_ALL}")
                break
            
            parsed_intent = parse_user_intent(user_input)
            intent = parsed_intent["intent"]
            problem_type = parsed_intent["problem_type"]
            
            if problem_type:
                print(f"\n{Fore.GREEN}Understood! Your intent is to '{Fore.YELLOW}{intent}{Fore.GREEN}' and the problem type is likely '{Fore.YELLOW}{problem_type}{Fore.GREEN}'.{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}Could not determine problem type from your input.{Style.RESET_ALL}")
                args.dataset = None
                continue
                
            print(f"\n{Fore.MAGENTA}" + "="*50)
            print("  MODEL SUGGESTION")
            print("="*50 + Style.RESET_ALL)

            suggested_models = get_suggested_models(problem_type)

            if suggested_models:
                if args.auto_run:
                    selected_model_class = suggested_models[0]['class']
                    print(f"Based on the problem type, I suggest using {selected_model_class.__name__}.")
                    print("\nAuto-run enabled. Running pipeline non-interactively...")
                    run_training_pipeline(handler, selected_model_class, problem_type, user_input, auto_run=True)
                else:
                    print(f"{Fore.CYAN}Based on the problem type, I suggest the following models:{Style.RESET_ALL}")
                    for i, model in enumerate(suggested_models):
                        print(f"  {Fore.YELLOW}{i + 1}{Style.RESET_ALL}. {Style.BRIGHT}{model['name']}{Style.RESET_ALL} - {model['description']}")

                    while True:
                        try:
                            choice = input(f"\n{Fore.YELLOW}Please select a model by number (1-{len(suggested_models)}) or type 'all' to run all: {Style.RESET_ALL}").strip()
                            if choice.lower() == 'all':
                                print("\nRunning all suggested models...")
                                all_results = []
                                for model in suggested_models:
                                    print(f"\n--- Running {model['name']} ---")
                                    results = run_training_pipeline(handler, model['class'], problem_type, user_input, auto_run=True)
                                    all_results.append((model['name'], results))
                                
                                # Print a summary table
                                print("\n" + "="*50)
                                print("  MODEL COMPARISON RESULTS")
                                print("="*50)
                                for name, results in all_results:
                                    print(f"\nModel: {name}")
                                    for metric, value in results.items():
                                        print(f"  {metric}: {value:.4f}")
                                print("="*50)
                                break
                            
                            selection = int(choice)
                            if 1 <= selection <= len(suggested_models):
                                selected_model_class = suggested_models[selection - 1]['class']
                                print(f"\nRunning pipeline with {selected_model_class.__name__}...")
                                run_training_pipeline(handler, selected_model_class, problem_type, user_input, auto_run=False)
                                break
                            else:
                                print(f"{Fore.RED}Invalid selection. Please enter a number between 1 and {len(suggested_models)}.{Style.RESET_ALL}")
                        except ValueError:
                            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}I don't have a model suggestion for this problem type yet. Let's explore other options.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Could not automatically detect the dataset type. Please provide more information.{Style.RESET_ALL}")
        
        choice = input(f"\n{Fore.CYAN}Task complete. Would you like to start a new task? (yes/no): {Style.RESET_ALL}").lower().strip()
        if choice not in ('yes', 'y'):
            print(f"\n{Fore.RED}Goodbye! 👋{Style.RESET_ALL}")
            break
        
        args.dataset = None

if __name__ == "__main__":
    main()