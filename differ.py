import difflib

with open('.\\preprocess_container\\druid_utils.py', 'r', encoding='utf-8') as file1:
    file1_lines = file1.readlines()

with open('.\\inference_container\\druid_utils.py', 'r', encoding='utf-8') as file2:
    file2_lines = file2.readlines()


# --- 3. Compute and display the difference ---
# difflib.unified_diff() returns a generator that yields the diff lines.
# We provide filenames for the header of the diff report.
diff = difflib.unified_diff(
    file1_lines,
    file2_lines,
    fromfile='file_v1.txt',
    tofile='file_v2.txt',
    lineterm='', # Prevents difflib from adding its own newlines
)

# We iterate through the generator and print each line of the diff.
print("--- Comparison Report ---")
for line in diff:
    print(line, end="") # The lines from the diff already include newlines