"""Presentation names only. Checkpoint contracts retain their original case IDs."""

METHODS = {
    0: dict(name='LR-YOLO gốc', title='Tái tạo toàn ảnh',
            detail='Nhánh phục hồi được tính trên toàn ảnh khi huấn luyện.'),
    6: dict(name='S1', title='Tái tạo chọn vùng',
            detail='Một vùng theo đối tượng và một vùng ngẫu nhiên; detection vẫn dùng toàn ảnh.'),
    8: dict(name='S2', title='Chọn vùng theo độ bao phủ',
            detail='Cải thiện độ bao phủ đối tượng từ tối đa 8 ứng viên; giữ nguyên ngân sách 2 vùng.'),
    9: dict(name='S3', title='S2 + bộ nhớ đặc trưng sạch',
            detail='Bổ sung ràng buộc đặc trưng theo prototype sạch và độ tin cậy khi huấn luyện. '
                   'Không cần bộ nhớ prototype khi demo suy luận.'),
}


def method_info(case_id, stage='dark'):
    if stage == 'clean':
        return dict(case_id=case_id, name='Clean', title='Checkpoint giai đoạn clean',
                    detail='Khởi tạo cho giai đoạn thích ứng, chưa phải kết quả cuối.')
    info = METHODS.get(case_id, dict(name=f'Case cũ {case_id}',
                                    title='Thử nghiệm trước đây',
                                    detail='Không thuộc ba phương án S1, S2, S3 trong báo cáo.'))
    return dict(case_id=case_id, **info)


def model_label(version, case_id, stage='dark'):
    return f'YOLO{version}-S · {method_info(case_id, stage)["name"]}'
