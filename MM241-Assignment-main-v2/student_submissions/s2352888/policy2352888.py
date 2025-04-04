from policy import Policy
import numpy as np

class Policy2352888(Policy):
    def __init__(self, policy_id=1):
        # Kiểm tra ID của policy, chỉ nhận giá trị 1 (FFD) hoặc 2 (BFD)
        assert policy_id in [1, 2], "Policy ID must be 1 (FFD) or 2 (BFD)"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        # Lấy danh sách sản phẩm và sắp xếp theo diện tích giảm dần
        sorted_products = sorted(
            observation["products"],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )

        for product in sorted_products:
            if product["quantity"] > 0:  # Chỉ xử lý nếu còn sản phẩm
                prod_size = product["size"]
                prod_w, prod_h = prod_size

                if self.policy_id == 1:  # First-Fit Decreasing
                    for stock_idx, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)

                        # Kiểm tra không xoay sản phẩm
                        if stock_w >= prod_w and stock_h >= prod_h:
                            position = self._find_first_fit_(stock, (prod_w, prod_h))
                            if position:
                                return {"stock_idx": stock_idx, "size": prod_size, "position": position}

                        # Kiểm tra xoay 90 độ
                        if stock_w >= prod_h and stock_h >= prod_w:
                            position = self._find_first_fit_(stock, (prod_h, prod_w))
                            if position:
                                return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": position}

                elif self.policy_id == 2:  # Best-Fit Decreasing
                    best_fit = None
                    best_fit_position = None
                    min_waste = float('inf')  # Giá trị thừa không gian nhỏ nhất

                    for stock_idx, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)

                        # Kiểm tra không xoay sản phẩm
                        if stock_w >= prod_w and stock_h >= prod_h:
                            position = self._find_first_fit_(stock, (prod_w, prod_h))
                            if position:
                                waste = (stock_w - prod_w) * (stock_h - prod_h)
                                if waste < min_waste:
                                    best_fit = (stock_idx, prod_size, position)
                                    min_waste = waste

                        # Kiểm tra xoay 90 độ
                        if stock_w >= prod_h and stock_h >= prod_w:
                            position = self._find_first_fit_(stock, (prod_h, prod_w))
                            if position:
                                waste = (stock_w - prod_h) * (stock_h - prod_w)
                                if waste < min_waste:
                                    best_fit = (stock_idx, prod_size[::-1], position)
                                    min_waste = waste

                    # Nếu tìm được tấm gỗ phù hợp nhất, trả về hành động
                    if best_fit:
                        stock_idx, prod_size, position = best_fit
                        return {"stock_idx": stock_idx, "size": prod_size, "position": position}

        # Nếu không tìm được vị trí phù hợp
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_first_fit_(self, stock, prod_size):
        """Tìm vị trí đầu tiên có thể đặt sản phẩm trong tấm gỗ."""
        prod_w, prod_h = prod_size
        stock_w, stock_h = stock.shape

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None
