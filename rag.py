dataset = []
with open('cat-facts.txt', 'r') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')