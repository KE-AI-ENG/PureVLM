
from purevlm.layer.qlinear import QLinear

def find_attr_path(root_obj, target_obj, root_name="model"):
    """
    在 root_obj 中递归查找 target_obj 的路径名
    支持对象、列表、元组、字典
    """
    visited = set()

    def _search(obj, path):
        if id(obj) in visited:
            return None
        visited.add(id(obj))

        if obj is target_obj:
            return path

        # 对象属性
        if hasattr(obj, "__dict__"):
            for attr_name, attr_value in obj.__dict__.items():
                sub_path = f"{path}.{attr_name}"
                result = _search(attr_value, sub_path)
                if result:
                    return result

        # 列表 / 元组
        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                sub_path = f"{path}[{idx}]"
                result = _search(item, sub_path)
                if result:
                    return result

        # 字典
        if isinstance(obj, dict):
            for key, value in obj.items():
                sub_path = f"{path}[{repr(key)}]"
                result = _search(value, sub_path)
                if result:
                    return result

        return None

    return _search(root_obj, root_name)

def get_obj_by_path(root_obj, path_str):
    current_obj = root_obj
    tokens = []
    i = 0
    while i < len(path_str):
        if path_str[i] == '.':
            i += 1
            start = i
            while i < len(path_str) and path_str[i] not in '.[':
                i += 1
            tokens.append(('attr', path_str[start:i]))
        elif path_str[i] == '[':
            i += 1
            start = i
            while i < len(path_str) and path_str[i] != ']':
                i += 1
            key_str = path_str[start:i]
            i += 1
            try:
                key = eval(key_str)
            except Exception:
                key = key_str
            tokens.append(('item', key))
        else:
            start = i
            while i < len(path_str) and path_str[i] not in '.[':
                i += 1
            tokens.append(('attr', path_str[start:i]))

    # 遍历 tokens 获取对象
    for typ, val in tokens:
        if typ == 'attr':
            # 如果是 list/tuple 并且 val 是数字字符串，按索引访问
            if isinstance(current_obj, (list, tuple)) and val.isdigit():
                current_obj = current_obj[int(val)]
            else:
                current_obj = getattr(current_obj, val)
        elif typ == 'item':
            current_obj = current_obj[val]

    return current_obj

def weight_loading(model, checkpoint, device='cuda'):
    loaded_keys = []
    failed_keys = []

    for key, tensor in checkpoint.items():
        try:
            device_tensor = tensor.to(device)

            if key.endswith(".weight") or key.endswith(".weight_packed") or key.endswith(".weight_shape") or key.endswith(".weight_scale"):
                layer_path = key.rsplit(".", 1)[0]
                layer_obj = get_obj_by_path(model, layer_path)

                if isinstance(layer_obj, QLinear):
                    layer_obj.set_weight(key, device_tensor)
                else:
                    layer_obj.weight = device_tensor
            else:
                parent_path, attr_name = key.rsplit(".", 1)
                parent_obj = get_obj_by_path(model, parent_path)
                setattr(parent_obj, attr_name, device_tensor)

            loaded_keys.append(key)

        except Exception as e:
            failed_keys.append((key, str(e)))

    return loaded_keys, failed_keys