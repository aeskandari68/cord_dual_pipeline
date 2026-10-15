"""Shared schema and CORD ground-truth adapter."""
from pydantic import BaseModel, Field


class Item(BaseModel):
    name: str
    price: str


class Receipt(BaseModel):
    items: list[Item] = Field(default_factory=list)
    total: str | None = None


def normalize_ground_truth(gt: dict) -> dict:
    """Map the subset of CORD gt_parse used for this small experiment."""
    menu = gt.get('menu', [])
    if isinstance(menu, dict):
        menu = [menu]
    if not isinstance(menu, list):
        menu = []
    items = [
        {'name': str(item['nm']), 'price': str(item['price']) if item.get('price') is not None else ''}
        for item in menu if isinstance(item, dict) and item.get('nm') is not None
    ]
    total = gt.get('total', {})
    if isinstance(total, list):
        total = next((x for x in total if isinstance(x, dict)), {})
    if not isinstance(total, dict):
        total = {}
    value = total.get('total_price')
    return {'items': items, 'total': str(value) if value is not None else None}
