import numpy as np

def read_npz(
    npz_path,
    max_elements=20,
    show_full=False
):
    """
    Simple NPZ reader.

    Args:
        npz_path (str): Path to the .npz file
        max_elements (int): Maximum number of elements to display per array
        show_full (bool): Whether to print the full array contents
    """
    data = np.load(npz_path, allow_pickle=True)

    print("=" * 60)
    print(f"NPZ file: {npz_path}")
    print(f"Number of arrays: {len(data.files)}")
    print("=" * 60)

    # -------- Basic information --------
    print("\n[Basic Information]")
    for key in data.files:
        arr = data[key]
        print(
            f"- key: {key}\n"
            f"  dtype: {arr.dtype}\n"
            f"  shape: {arr.shape}\n"
            f"  size: {arr.size}"
        )

    # -------- Array contents --------
    print("\n[Array Contents]")
    for key in data.files:
        arr = data[key]
        print("-" * 60)
        print(f"key: {key}")

        if show_full:
            print(arr)
        else:
            flat = arr.flatten()
            if flat.size <= max_elements:
                print(flat.reshape(arr.shape))
            else:
                print(f"First {max_elements} elements:")
                print(flat[:max_elements])
                print("... (remaining elements omitted)")

    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    read_npz(
        "/path/to/npz",
        max_elements=10,
        show_full=False
    )
