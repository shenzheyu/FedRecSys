import torch


class LinearRegression(torch.nn.Module):
    """
    A pytorch implementation of LinearRegression Model.
    """

    def __init__(self, categorical_field_dims, numerical_num):
        super().__init__()
        print(
            "categorical_field_dims = {}, numerical_num = {}".format(
                categorical_field_dims, numerical_num
            )
        )
        # len(categorical_field_dims) = 24, numerical_num = 2
        self.categorical_field_dims = categorical_field_dims
        self.linear_layer = torch.nn.Linear(numerical_num + len(categorical_field_dims), 1)
        self.out_unit = torch.nn.Sigmoid()

    def forward(self, categorical_x, numerical_x):
        """
        :param
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)`` # [2048, 24])
        numerical_x: Long tensor of size ``(batch_size, numerical_num)`` # [2048, 2]
        """

        logit = self.linear_layer(torch.cat([(categorical_x / self.categorical_field_dims).float(), numerical_x], 1)) # [2048, 1]
        return [self.out_unit(logit)]
