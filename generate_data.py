"""
Generate synthetic Vietnamese conversations cho testing pipeline.

Tạo thêm ~35 hội thoại (đa dạng kịch bản lừa đảo + hội thoại bình thường)
để kết hợp với 30 hội thoại có sẵn trong raw_conversations.json.
Output: streaming/data/synthetic_conversations.json
"""

import json
import os
import random

STREAMING_ROOT = os.path.dirname(os.path.abspath(__file__))


def build_scam_conversations():
    """Tạo hội thoại SCAM đa dạng kịch bản."""
    return [
        # ── 1. Crypto investment scam ──
        {
            "conversation_id": "synth_s01",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s01_t01", "speaker_role": "scammer",
                 "text": "Chào bạn, bạn có muốn đầu tư Bitcoin với lãi suất 500% trong 30 ngày không?",
                 "t4_labels": ["LIKING", "COMMITMENT"]},
                {"turn_id": "synth_s01_t02", "speaker_role": "normal",
                 "text": "Nghe hấp dẫn nhỉ, nhưng làm sao được lãi cao vậy?"},
                {"turn_id": "synth_s01_t03", "speaker_role": "scammer",
                 "text": "Chúng tôi có đội ngũ chuyên gia giao dịch AI tự động. Bạn chỉ cần nạp tối thiểu 5 triệu vào ví điện tử, hệ thống sẽ tự sinh lời.",
                 "t4_labels": ["AUTHORITY", "ACTION_REQUEST"]},
                {"turn_id": "synth_s01_t04", "speaker_role": "normal",
                 "text": "Có an toàn không? Rút tiền được không?"},
                {"turn_id": "synth_s01_t05", "speaker_role": "scammer",
                 "text": "Hoàn toàn an toàn, đã có hàng nghìn người tham gia thành công. Nhưng chương trình chỉ mở thêm 24 giờ nữa thôi, bạn nên nhanh.",
                 "t4_labels": ["SOCIAL_PROOF", "SCARCITY"]},
            ],
        },
        # ── 2. Bank card expiry scam ──
        {
            "conversation_id": "synth_s02",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s02_t01", "speaker_role": "normal",
                 "text": "Alo ai đấy ạ?"},
                {"turn_id": "synth_s02_t02", "speaker_role": "scammer",
                 "text": "Xin chào, tôi gọi từ ngân hàng BIDV. Thẻ ATM của bạn sắp hết hạn và cần gia hạn trực tuyến ngay hôm nay.",
                 "t4_labels": ["AUTHORITY", "SCARCITY"]},
                {"turn_id": "synth_s02_t03", "speaker_role": "normal",
                 "text": "Thẻ tôi vẫn dùng được bình thường mà?"},
                {"turn_id": "synth_s02_t04", "speaker_role": "scammer",
                 "text": "Hệ thống mới cập nhật ạ. Bạn cung cấp số thẻ, ngày hết hạn và mã CVV để em xử lý nhé.",
                 "t4_labels": ["INFO_REQUEST", "DEFLECT_MINIMIZE"]},
            ],
        },
        # ── 3. Fake relative accident ──
        {
            "conversation_id": "synth_s03",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s03_t01", "speaker_role": "scammer",
                 "text": "Chào chị, em gọi từ bệnh viện Chợ Rẫy. Chồng chị vừa bị tai nạn giao thông nghiêm trọng và đang cấp cứu.",
                 "t4_labels": ["AUTHORITY", "OVERWHELM"]},
                {"turn_id": "synth_s03_t02", "speaker_role": "normal",
                 "text": "Trời ơi! Anh ấy bị sao rồi?"},
                {"turn_id": "synth_s03_t03", "speaker_role": "scammer",
                 "text": "Bác sĩ đang mổ khẩn cấp. Chị cần chuyển 50 triệu viện phí trong vòng 1 giờ, nếu không bệnh viện sẽ ngưng điều trị.",
                 "t4_labels": ["SCARCITY", "ACTION_REQUEST", "THREAT_FINANCIAL"]},
                {"turn_id": "synth_s03_t04", "speaker_role": "normal",
                 "text": "Chuyển vào đâu?"},
                {"turn_id": "synth_s03_t05", "speaker_role": "scammer",
                 "text": "Tài khoản Vietcombank 0371xxxxxx mang tên Nguyễn Văn Tâm, đây là tài khoản quỹ cấp cứu.",
                 "t4_labels": ["ACTION_REQUEST"]},
            ],
        },
        # ── 4. Tax refund scam ──
        {
            "conversation_id": "synth_s04",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s04_t01", "speaker_role": "normal",
                 "text": "Alo xin nghe?"},
                {"turn_id": "synth_s04_t02", "speaker_role": "scammer",
                 "text": "Chào anh, tôi gọi từ Cục Thuế. Anh được hoàn thuế thu nhập cá nhân 7 triệu đồng cho năm 2025.",
                 "t4_labels": ["AUTHORITY", "LIKING"]},
                {"turn_id": "synth_s04_t03", "speaker_role": "normal",
                 "text": "Thật hả? Tôi phải làm gì?"},
                {"turn_id": "synth_s04_t04", "speaker_role": "scammer",
                 "text": "Anh chỉ cần cung cấp số tài khoản ngân hàng và mã OTP gửi đến điện thoại để xác nhận nhận tiền.",
                 "t4_labels": ["INFO_REQUEST", "ACTION_REQUEST"]},
            ],
        },
        # ── 5. Online shopping refund scam ──
        {
            "conversation_id": "synth_s05",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s05_t01", "speaker_role": "scammer",
                 "text": "Chào bạn, đây là bộ phận hỗ trợ Lazada. Đơn hàng LZ38291 của bạn bị lỗi thanh toán kép và bạn được hoàn 2 lần tiền.",
                 "t4_labels": ["AUTHORITY", "LIKING"]},
                {"turn_id": "synth_s05_t02", "speaker_role": "normal",
                 "text": "Ồ tôi có đơn đó thật. Hoàn kiểu gì?"},
                {"turn_id": "synth_s05_t03", "speaker_role": "scammer",
                 "text": "Bạn vào link mà em gửi qua tin nhắn, đăng nhập tài khoản ngân hàng để nhận hoàn tiền. Link chỉ có hiệu lực 30 phút.",
                 "t4_labels": ["ACTION_REQUEST", "SCARCITY"]},
            ],
        },
        # ── 6. Loan approval scam ──
        {
            "conversation_id": "synth_s06",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s06_t01", "speaker_role": "scammer",
                 "text": "Xin chào, bạn đã được duyệt khoản vay tín chấp 200 triệu với lãi suất 0% trong 12 tháng.",
                 "t4_labels": ["LIKING"]},
                {"turn_id": "synth_s06_t02", "speaker_role": "normal",
                 "text": "Tôi có đăng ký vay đâu nhỉ?"},
                {"turn_id": "synth_s06_t03", "speaker_role": "scammer",
                 "text": "Đây là chương trình ưu đãi tự động của ngân hàng dành cho khách hàng có điểm tín dụng cao. Bạn chỉ cần đóng phí giải ngân 3 triệu.",
                 "t4_labels": ["AUTHORITY", "ACTION_REQUEST", "RECIPROCITY"]},
                {"turn_id": "synth_s06_t04", "speaker_role": "normal",
                 "text": "Phí giải ngân là sao?"},
                {"turn_id": "synth_s06_t05", "speaker_role": "scammer",
                 "text": "Phí xử lý hồ sơ theo quy định. Sau khi đóng, tiền vay sẽ được chuyển vào tài khoản trong 30 phút. Hạn cuối hôm nay.",
                 "t4_labels": ["DEFLECT_MINIMIZE", "SCARCITY"]},
            ],
        },
        # ── 7. Social media hacked ──
        {
            "conversation_id": "synth_s07",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s07_t01", "speaker_role": "normal",
                 "text": "Alo?"},
                {"turn_id": "synth_s07_t02", "speaker_role": "scammer",
                 "text": "Chào bạn, tôi là nhân viên bảo mật Facebook. Tài khoản Facebook của bạn đang bị hack và có người đang sử dụng để lừa đảo.",
                 "t4_labels": ["AUTHORITY", "OVERWHELM"]},
                {"turn_id": "synth_s07_t03", "speaker_role": "normal",
                 "text": "Thật hả? Tôi vẫn đăng nhập được bình thường mà?"},
                {"turn_id": "synth_s07_t04", "speaker_role": "scammer",
                 "text": "Hacker đang dùng phiên đăng nhập song song. Bạn cần cung cấp mật khẩu và mã xác minh 2 bước để chúng tôi khóa phiên trái phép.",
                 "t4_labels": ["INFO_REQUEST", "HELP"]},
            ],
        },
        # ── 8. Fake scholarship ──
        {
            "conversation_id": "synth_s08",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s08_t01", "speaker_role": "scammer",
                 "text": "Chào em, em được chọn nhận học bổng toàn phần du học Nhật Bản trị giá 500 triệu từ quỹ giáo dục ABC.",
                 "t4_labels": ["LIKING", "AUTHORITY"]},
                {"turn_id": "synth_s08_t02", "speaker_role": "normal",
                 "text": "Em không nhớ đã đăng ký chương trình nào cả."},
                {"turn_id": "synth_s08_t03", "speaker_role": "scammer",
                 "text": "Đây là chương trình xét tuyển tự động dựa trên hồ sơ học tập. Em cần đóng phí xử lý hồ sơ 5 triệu trước ngày mai để giữ suất.",
                 "t4_labels": ["SCARCITY", "ACTION_REQUEST"]},
            ],
        },
        # ── 9. Water bill fraud ──
        {
            "conversation_id": "synth_s09",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s09_t01", "speaker_role": "normal",
                 "text": "Xin chào ai gọi ạ?"},
                {"turn_id": "synth_s09_t02", "speaker_role": "scammer",
                 "text": "Đây là Công ty Cấp thoát nước TP.HCM. Hộ gia đình bạn đang nợ tiền nước 3 tháng tổng cộng 12 triệu và sẽ bị cắt nước trong hôm nay.",
                 "t4_labels": ["AUTHORITY", "THREAT_FINANCIAL", "SCARCITY"]},
                {"turn_id": "synth_s09_t03", "speaker_role": "normal",
                 "text": "Không thể nào, tôi trả đủ mỗi tháng rồi mà."},
                {"turn_id": "synth_s09_t04", "speaker_role": "scammer",
                 "text": "Có thể do hệ thống cập nhật chậm. Bạn thanh toán ngay qua tài khoản 0291xxxxxx để tránh bị cắt, chúng tôi sẽ kiểm tra sau.",
                 "t4_labels": ["DEFLECT_EXTERNAL", "ACTION_REQUEST"]},
            ],
        },
        # ── 10. Fake charity ──
        {
            "conversation_id": "synth_s10",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s10_t01", "speaker_role": "scammer",
                 "text": "Chào anh chị, tôi đại diện Quỹ Hy Vọng kêu gọi quyên góp cho trẻ em vùng lũ miền Trung. Mỗi suất ăn chỉ 50 nghìn đồng.",
                 "t4_labels": ["AUTHORITY", "LIKING"]},
                {"turn_id": "synth_s10_t02", "speaker_role": "normal",
                 "text": "Tôi muốn ủng hộ nhưng sao biết đây là quỹ thật?"},
                {"turn_id": "synth_s10_t03", "speaker_role": "scammer",
                 "text": "Anh yên tâm, quỹ được Bộ LĐTBXH công nhận. Anh chuyển khoản vào số tài khoản cá nhân 0412xxxxxx, tôi sẽ gửi biên lai qua Zalo.",
                 "t4_labels": ["SOCIAL_PROOF", "ACTION_REQUEST", "DEFLECT_MINIMIZE"]},
            ],
        },
        # ── 11. Fake customs package ──
        {
            "conversation_id": "synth_s11",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s11_t01", "speaker_role": "normal",
                 "text": "Alo ai gọi đấy?"},
                {"turn_id": "synth_s11_t02", "speaker_role": "scammer",
                 "text": "Chào bạn, đây là hải quan sân bay Tân Sơn Nhất. Bạn có kiện hàng từ Mỹ đang bị giữ do chưa nộp thuế nhập khẩu.",
                 "t4_labels": ["AUTHORITY", "THREAT_LEGAL"]},
                {"turn_id": "synth_s11_t03", "speaker_role": "normal",
                 "text": "Tôi không có đặt hàng gì từ Mỹ cả."},
                {"turn_id": "synth_s11_t04", "speaker_role": "scammer",
                 "text": "Kiện hàng ghi tên và địa chỉ của bạn. Nếu không thanh toán thuế 8 triệu trong ngày, kiện hàng sẽ bị tịch thu và bạn sẽ bị phạt hành chính.",
                 "t4_labels": ["SCARCITY", "THREAT_LEGAL", "ACTION_REQUEST"]},
            ],
        },
        # ── 12. Fake tech support ──
        {
            "conversation_id": "synth_s12",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s12_t01", "speaker_role": "scammer",
                 "text": "Chào bạn, tôi là kỹ thuật viên Microsoft. Máy tính của bạn đã bị nhiễm virus nghiêm trọng và đang gửi dữ liệu ra ngoài.",
                 "t4_labels": ["AUTHORITY", "OVERWHELM"]},
                {"turn_id": "synth_s12_t02", "speaker_role": "normal",
                 "text": "Sao bạn biết máy tôi bị virus?"},
                {"turn_id": "synth_s12_t03", "speaker_role": "scammer",
                 "text": "Hệ thống bảo mật Microsoft phát hiện qua IP của bạn. Bạn cần cài phần mềm diệt virus của chúng tôi ngay, phí 2 triệu đồng.",
                 "t4_labels": ["DEFLECT_EXTERNAL", "SCARCITY", "ACTION_REQUEST"]},
            ],
        },
        # ── 13. Inheritance scam ──
        {
            "conversation_id": "synth_s13",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s13_t01", "speaker_role": "scammer",
                 "text": "Xin chào, tôi là luật sư Trần Hải từ văn phòng công chứng quốc tế. Bạn có người thân ở Đức vừa qua đời và để lại di sản 2 tỷ đồng cho bạn.",
                 "t4_labels": ["AUTHORITY", "LIKING"]},
                {"turn_id": "synth_s13_t02", "speaker_role": "normal",
                 "text": "Tôi có ai ở Đức đâu?"},
                {"turn_id": "synth_s13_t03", "speaker_role": "scammer",
                 "text": "Theo hồ sơ di chúc, bạn là người thừa kế hợp pháp. Để nhận tiền, bạn cần đóng phí luật sư và thuế chuyển nhượng tổng 15 triệu.",
                 "t4_labels": ["DEFLECT_MINIMIZE", "ACTION_REQUEST"]},
                {"turn_id": "synth_s13_t04", "speaker_role": "normal",
                 "text": "Sao phải đóng trước vậy?"},
                {"turn_id": "synth_s13_t05", "speaker_role": "scammer",
                 "text": "Đây là quy định pháp luật quốc tế, sau khi nhận tiền bạn sẽ được hoàn lại. Hạn cuối xử lý hồ sơ là ngày mai.",
                 "t4_labels": ["DEFLECT_EXTERNAL", "SCARCITY"]},
            ],
        },
        # ── 14. Electricity new app scam ──
        {
            "conversation_id": "synth_s14",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s14_t01", "speaker_role": "scammer",
                 "text": "Chào quý khách, Điện lực Việt Nam thông báo bắt buộc chuyển sang thanh toán qua ứng dụng mới EVN Smart. Vui lòng tải app qua link sau.",
                 "t4_labels": ["AUTHORITY", "SCARCITY"]},
                {"turn_id": "synth_s14_t02", "speaker_role": "normal",
                 "text": "Tôi đang dùng app EVN cũ mà sao phải đổi?"},
                {"turn_id": "synth_s14_t03", "speaker_role": "scammer",
                 "text": "App cũ sẽ ngừng hoạt động từ đầu tháng sau. Nếu không cài app mới, hóa đơn sẽ bị tính phí trễ 20%. Bạn tải và đăng nhập bằng tài khoản ngân hàng.",
                 "t4_labels": ["THREAT_FINANCIAL", "ACTION_REQUEST"]},
            ],
        },
        # ── 15. Romance scam ──
        {
            "conversation_id": "synth_s15",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s15_t01", "speaker_role": "normal",
                 "text": "Chào bạn, bạn gửi tin nhắn cho tôi trên Zalo à?"},
                {"turn_id": "synth_s15_t02", "speaker_role": "scammer",
                 "text": "Ừ, anh kết bạn vì thấy avatar em dễ thương. Anh là việt kiều Mỹ, đang tìm bạn tâm giao.",
                 "t4_labels": ["LIKING"]},
                {"turn_id": "synth_s15_t03", "speaker_role": "normal",
                 "text": "Cảm ơn nhưng tôi không quen biết anh."},
                {"turn_id": "synth_s15_t04", "speaker_role": "scammer",
                 "text": "Anh muốn gửi quà cho em từ Mỹ, một chiếc túi xách hàng hiệu. Nhưng bưu kiện bị giữ ở hải quan, em cần đóng phí 3 triệu để nhận.",
                 "t4_labels": ["LIKING", "ACTION_REQUEST", "RECIPROCITY"]},
            ],
        },
        # ── 16. Fake police warrant ──
        {
            "conversation_id": "synth_s16",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s16_t01", "speaker_role": "normal",
                 "text": "Ai gọi đấy ạ?"},
                {"turn_id": "synth_s16_t02", "speaker_role": "scammer",
                 "text": "Tôi là Trung tá Phạm Quốc Tuấn, Bộ Công an. Bạn đang có lệnh truy nã quốc tế vì liên quan giao dịch bất hợp pháp.",
                 "t4_labels": ["AUTHORITY", "THREAT_LEGAL", "OVERWHELM"]},
                {"turn_id": "synth_s16_t03", "speaker_role": "normal",
                 "text": "Không thể nào, tôi là công dân bình thường!"},
                {"turn_id": "synth_s16_t04", "speaker_role": "scammer",
                 "text": "Để gỡ lệnh truy nã, bạn cần chuyển tài sản vào tài khoản tạm giữ của cơ quan điều tra. Không ai khác được biết chuyện này.",
                 "t4_labels": ["ACTION_REQUEST", "ISOLATION"]},
            ],
        },
        # ── 17. Fake insurance refund ──
        {
            "conversation_id": "synth_s17",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s17_t01", "speaker_role": "scammer",
                 "text": "Chào chị, đây là bảo hiểm nhân thọ Manulife. Hợp đồng bảo hiểm của chị đủ điều kiện nhận tiền đáo hạn 80 triệu.",
                 "t4_labels": ["AUTHORITY", "LIKING"]},
                {"turn_id": "synth_s17_t02", "speaker_role": "normal",
                 "text": "Tôi có mua bảo hiểm thật. Nhưng chưa đến hạn mà?"},
                {"turn_id": "synth_s17_t03", "speaker_role": "scammer",
                 "text": "Đây là chương trình thanh toán sớm đặc biệt. Chị cung cấp số CMND, số tài khoản và chụp mặt trước thẻ ngân hàng để xác nhận.",
                 "t4_labels": ["INFO_REQUEST", "DEFLECT_MINIMIZE"]},
                {"turn_id": "synth_s17_t04", "speaker_role": "normal",
                 "text": "Tôi sẽ gọi lại tổng đài Manulife kiểm tra."},
                {"turn_id": "synth_s17_t05", "speaker_role": "scammer",
                 "text": "Không cần đâu chị, chương trình hết hạn trong 2 giờ nữa. Nếu chị không xác nhận ngay sẽ mất quyền lợi.",
                 "t4_labels": ["SCARCITY", "THREAT_FINANCIAL"]},
            ],
        },
        # ── 18. Fake Zalo OTP ──
        {
            "conversation_id": "synth_s18",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s18_t01", "speaker_role": "normal",
                 "text": "Alo?"},
                {"turn_id": "synth_s18_t02", "speaker_role": "scammer",
                 "text": "Em ơi, anh Tuấn đây, bạn cùng công ty. Anh đang lỡ tay đăng xuất Zalo, em gửi giúp anh mã OTP vừa gửi đến điện thoại em được không?",
                 "t4_labels": ["LIKING", "INFO_REQUEST"]},
                {"turn_id": "synth_s18_t03", "speaker_role": "normal",
                 "text": "Anh Tuấn nào? Sao mã OTP lại gửi đến máy em?"},
                {"turn_id": "synth_s18_t04", "speaker_role": "scammer",
                 "text": "Anh nhớ nhầm số thôi mà. Gửi nhanh giúp anh, đang cần gấp để gửi file cho sếp.",
                 "t4_labels": ["SCARCITY", "DEFLECT_MINIMIZE"]},
            ],
        },
        # ── 19. Fake apartment rental ──
        {
            "conversation_id": "synth_s19",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s19_t01", "speaker_role": "normal",
                 "text": "Anh ơi, em hỏi về phòng trọ đăng trên Batdongsan.com, còn không ạ?"},
                {"turn_id": "synth_s19_t02", "speaker_role": "scammer",
                 "text": "Còn em ơi, phòng đẹp lắm, 25m2, gần trung tâm, giá chỉ 2 triệu tháng bao điện nước.",
                 "t4_labels": ["LIKING"]},
                {"turn_id": "synth_s19_t03", "speaker_role": "normal",
                 "text": "Rẻ vậy á? Em muốn đến xem phòng."},
                {"turn_id": "synth_s19_t04", "speaker_role": "scammer",
                 "text": "Em đặt cọc trước 3 triệu qua chuyển khoản để giữ phòng nha, nhiều người hỏi lắm. Xem phòng thì cuối tuần.",
                 "t4_labels": ["SCARCITY", "ACTION_REQUEST"]},
            ],
        },
        # ── 20. Fake e-commerce job ──
        {
            "conversation_id": "synth_s20",
            "t1_label": "SCAM",
            "messages": [
                {"turn_id": "synth_s20_t01", "speaker_role": "scammer",
                 "text": "Chào bạn, đây là nhóm việc làm Tiki. Bạn có muốn kiếm thêm 3-8 triệu mỗi ngày bằng cách đánh giá sản phẩm không?",
                 "t4_labels": ["AUTHORITY", "LIKING"]},
                {"turn_id": "synth_s20_t02", "speaker_role": "normal",
                 "text": "Cụ thể phải làm gì vậy?"},
                {"turn_id": "synth_s20_t03", "speaker_role": "scammer",
                 "text": "Bạn mua sản phẩm trên Tiki và để lại đánh giá 5 sao. Tiền mua hàng sẽ được hoàn lại cùng hoa hồng. Nhiệm vụ đầu miễn phí, sau đó cần ứng trước.",
                 "t4_labels": ["COMMITMENT", "RECIPROCITY", "ACTION_REQUEST"]},
                {"turn_id": "synth_s20_t04", "speaker_role": "normal",
                 "text": "Nghe có vẻ không an toàn."},
                {"turn_id": "synth_s20_t05", "speaker_role": "scammer",
                 "text": "Hàng chục nghìn người đã tham gia thành công rồi. Nhóm Telegram có đầy bằng chứng rút tiền. Bạn không thử sẽ tiếc đấy!",
                 "t4_labels": ["SOCIAL_PROOF", "SCARCITY"]},
            ],
        },
    ]


def build_ambiguous_conversations():
    """Tạo hội thoại AMBIGUOUS – có dấu hiệu đáng ngờ nhưng chưa rõ ràng."""
    return [
        # ── 1. Suspicious survey call ──
        {
            "conversation_id": "synth_a01",
            "t1_label": "AMBIGUOUS",
            "messages": [
                {"turn_id": "synth_a01_t01", "speaker_role": "normal",
                 "text": "Alo xin nghe?"},
                {"turn_id": "synth_a01_t02", "speaker_role": "scammer",
                 "text": "Chào anh, em gọi từ trung tâm khảo sát Nielsen. Em muốn hỏi anh vài câu về thói quen tiêu dùng, chỉ mất 5 phút.",
                 "t4_labels": ["AUTHORITY"]},
                {"turn_id": "synth_a01_t03", "speaker_role": "normal",
                 "text": "Ừ được, hỏi đi."},
                {"turn_id": "synth_a01_t04", "speaker_role": "scammer",
                 "text": "Cuối cùng, anh cho em xin email và số CMND để gửi quà cảm ơn nhé.",
                 "t4_labels": ["INFO_REQUEST"]},
            ],
        },
        # ── 2. Suspicious health product ──
        {
            "conversation_id": "synth_a02",
            "t1_label": "AMBIGUOUS",
            "messages": [
                {"turn_id": "synth_a02_t01", "speaker_role": "scammer",
                 "text": "Chào chị, em là tư vấn viên sức khỏe. Chị có biết sản phẩm collagen Nhật Bản đang giảm 70% cho 50 khách đầu tiên không?",
                 "t4_labels": ["SCARCITY", "LIKING"]},
                {"turn_id": "synth_a02_t02", "speaker_role": "normal",
                 "text": "Tôi không đăng ký tư vấn."},
                {"turn_id": "synth_a02_t03", "speaker_role": "scammer",
                 "text": "Dạ em có số chị từ hệ thống khách hàng VIP. Sản phẩm cam kết hiệu quả 100%, thanh toán khi nhận hàng ạ.",
                 "t4_labels": ["DEFLECT_MINIMIZE"]},
            ],
        },
        # ── 3. Suspicious headhunter ──
        {
            "conversation_id": "synth_a03",
            "t1_label": "AMBIGUOUS",
            "messages": [
                {"turn_id": "synth_a03_t01", "speaker_role": "scammer",
                 "text": "Chào anh, em từ công ty tuyển dụng TopCV. Hồ sơ anh rất phù hợp vị trí quản lý lương 50 triệu.",
                 "t4_labels": ["AUTHORITY", "LIKING"]},
                {"turn_id": "synth_a03_t02", "speaker_role": "normal",
                 "text": "Tôi đúng là có đăng CV trên TopCV. Vị trí ở công ty nào?"},
                {"turn_id": "synth_a03_t03", "speaker_role": "scammer",
                 "text": "Em chưa tiện tiết lộ tên công ty, nhưng anh tham gia buổi phỏng vấn online qua link em gửi nhé. Anh cần tải app Zoom đặc biệt.",
                 "t4_labels": ["ACTION_REQUEST", "DEFLECT_EXTERNAL"]},
            ],
        },
        # ── 4. Credit card offer ──
        {
            "conversation_id": "synth_a04",
            "t1_label": "AMBIGUOUS",
            "messages": [
                {"turn_id": "synth_a04_t01", "speaker_role": "normal",
                 "text": "Xin chào?"},
                {"turn_id": "synth_a04_t02", "speaker_role": "scammer",
                 "text": "Chào anh, em từ ngân hàng Techcombank. Anh được ưu đãi mở thẻ tín dụng platinum miễn phí năm đầu.",
                 "t4_labels": ["AUTHORITY", "LIKING"]},
                {"turn_id": "synth_a04_t03", "speaker_role": "normal",
                 "text": "Tôi đang dùng thẻ TCB rồi, nhưng thôi khỏi mở thêm."},
                {"turn_id": "synth_a04_t04", "speaker_role": "scammer",
                 "text": "Hạn mức thẻ mới cao gấp 3 lần anh ạ. Em chỉ cần xác nhận thông tin thẻ cũ để nâng cấp tự động.",
                 "t4_labels": ["INFO_REQUEST"]},
            ],
        },
        # ── 5. Suspicious shopping discount ──
        {
            "conversation_id": "synth_a05",
            "t1_label": "AMBIGUOUS",
            "messages": [
                {"turn_id": "synth_a05_t01", "speaker_role": "scammer",
                 "text": "Chào bạn, shop Minh Tâm đang giảm giá iPhone 16 Pro Max chỉ 15 triệu, chính hãng bảo hành 12 tháng.",
                 "t4_labels": ["LIKING"]},
                {"turn_id": "synth_a05_t02", "speaker_role": "normal",
                 "text": "Sao rẻ vậy? Giá thị trường 30 triệu mà."},
                {"turn_id": "synth_a05_t03", "speaker_role": "scammer",
                 "text": "Hàng xách tay số lượng có hạn. Đặt cọc 5 triệu giữ máy, ship COD toàn quốc.",
                 "t4_labels": ["SCARCITY", "ACTION_REQUEST"]},
            ],
        },
    ]


def build_legit_conversations():
    """Tạo hội thoại LEGIT – bình thường, đời thường."""
    return [
        # ── 1. Pizza delivery ──
        {
            "conversation_id": "synth_l01",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l01_t01", "speaker_role": "normal",
                 "text": "Alo Pizza Hut phải không ạ?"},
                {"turn_id": "synth_l01_t02", "speaker_role": "normal",
                 "text": "Dạ đúng rồi ạ. Anh chị muốn đặt gì ạ?"},
                {"turn_id": "synth_l01_t03", "speaker_role": "normal",
                 "text": "Cho tôi 1 pizza hải sản cỡ lớn và 2 lon Pepsi nhé."},
                {"turn_id": "synth_l01_t04", "speaker_role": "normal",
                 "text": "Dạ tổng 285 nghìn ạ. Anh cho em địa chỉ giao hàng."},
                {"turn_id": "synth_l01_t05", "speaker_role": "normal",
                 "text": "Số 12 Lê Văn Sỹ, phường 1, Tân Bình nhé."},
            ],
        },
        # ── 2. Gym trainer scheduling ──
        {
            "conversation_id": "synth_l02",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l02_t01", "speaker_role": "normal",
                 "text": "Coach ơi, mai mình tập lúc mấy giờ ạ?"},
                {"turn_id": "synth_l02_t02", "speaker_role": "normal",
                 "text": "6 giờ sáng nha bạn. Nhớ ăn sáng nhẹ trước 1 tiếng."},
                {"turn_id": "synth_l02_t03", "speaker_role": "normal",
                 "text": "Dạ vâng, mai tập bài gì vậy coach?"},
                {"turn_id": "synth_l02_t04", "speaker_role": "normal",
                 "text": "Ngày mai tập chân và core nhé. Mang giày đế phẳng."},
            ],
        },
        # ── 3. Parent-teacher discussion ──
        {
            "conversation_id": "synth_l03",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l03_t01", "speaker_role": "normal",
                 "text": "Cô giáo ơi, con em học kỳ này học lực thế nào ạ?"},
                {"turn_id": "synth_l03_t02", "speaker_role": "normal",
                 "text": "Cháu Minh học tốt lắm chị ạ, đặc biệt môn Toán tiến bộ nhiều."},
                {"turn_id": "synth_l03_t03", "speaker_role": "normal",
                 "text": "Dạ cảm ơn cô, con em có cần bổ sung môn nào không ạ?"},
                {"turn_id": "synth_l03_t04", "speaker_role": "normal",
                 "text": "Cháu nên luyện thêm tiếng Anh, đặc biệt kỹ năng nghe. Chị cho cháu nghe podcast mỗi ngày thì tốt."},
            ],
        },
        # ── 4. Dentist appointment ──
        {
            "conversation_id": "synth_l04",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l04_t01", "speaker_role": "normal",
                 "text": "Chào bác sĩ, em muốn đặt lịch khám răng."},
                {"turn_id": "synth_l04_t02", "speaker_role": "normal",
                 "text": "Chào em, bạn muốn khám vào ngày nào?"},
                {"turn_id": "synth_l04_t03", "speaker_role": "normal",
                 "text": "Thứ 7 tuần này được không ạ? Buổi sáng."},
                {"turn_id": "synth_l04_t04", "speaker_role": "normal",
                 "text": "Được em nhé, 9 giờ sáng thứ 7. Em nhớ không ăn gì 2 tiếng trước khi khám."},
            ],
        },
        # ── 5. Neighbor help request ──
        {
            "conversation_id": "synth_l05",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l05_t01", "speaker_role": "normal",
                 "text": "Chị Hương ơi, nhà em bị mất điện, bên chị có bị không ạ?"},
                {"turn_id": "synth_l05_t02", "speaker_role": "normal",
                 "text": "Nhà chị cũng mất rồi em ơi. Chắc cả khu bị."},
                {"turn_id": "synth_l05_t03", "speaker_role": "normal",
                 "text": "Chị có số tổng đài điện lực không ạ?"},
                {"turn_id": "synth_l05_t04", "speaker_role": "normal",
                 "text": "1900 6969 em nhé. Chị gọi rồi họ nói sửa trong 2 tiếng."},
            ],
        },
        # ── 6. Birthday party planning ──
        {
            "conversation_id": "synth_l06",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l06_t01", "speaker_role": "normal",
                 "text": "Ê mày, sinh nhật tao tuần sau, tới nhé!"},
                {"turn_id": "synth_l06_t02", "speaker_role": "normal",
                 "text": "OK chứ! Mấy giờ và ở đâu?"},
                {"turn_id": "synth_l06_t03", "speaker_role": "normal",
                 "text": "7 giờ tối thứ 7, quán BBQ trên Trần Hưng Đạo. Tao đặt bàn 15 người rồi."},
                {"turn_id": "synth_l06_t04", "speaker_role": "normal",
                 "text": "Ngon! Tao mang theo bánh kem nhé."},
                {"turn_id": "synth_l06_t05", "speaker_role": "normal",
                 "text": "Khỏi khỏi, tao đặt rồi. Mày đến thôi."},
            ],
        },
        # ── 7. School pickup coordination ──
        {
            "conversation_id": "synth_l07",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l07_t01", "speaker_role": "normal",
                 "text": "Anh ơi hôm nay anh đón con được không? Em họp đến tận 6 giờ."},
                {"turn_id": "synth_l07_t02", "speaker_role": "normal",
                 "text": "Được em, mấy giờ con tan?"},
                {"turn_id": "synth_l07_t03", "speaker_role": "normal",
                 "text": "4 giờ 30 anh nhé. Nhớ cho con ăn xế nha, sáng con không ăn mấy."},
                {"turn_id": "synth_l07_t04", "speaker_role": "normal",
                 "text": "OK em, anh mua bánh mì cho con. Yên tâm đi."},
            ],
        },
        # ── 8. Recipe sharing ──
        {
            "conversation_id": "synth_l08",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l08_t01", "speaker_role": "normal",
                 "text": "Chị ơi, chị nấu phở bò ngon quá, cho em xin công thức được không?"},
                {"turn_id": "synth_l08_t02", "speaker_role": "normal",
                 "text": "Được chứ em. Quan trọng nhất là nước dùng phải ninh xương ống 6-8 tiếng."},
                {"turn_id": "synth_l08_t03", "speaker_role": "normal",
                 "text": "Gia vị thì chị dùng gì ạ?"},
                {"turn_id": "synth_l08_t04", "speaker_role": "normal",
                 "text": "Hồi, quế, thảo quả, gừng nướng. Chị gửi công thức chi tiết qua Zalo nhé."},
            ],
        },
        # ── 9. Car repair inquiry ──
        {
            "conversation_id": "synth_l09",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l09_t01", "speaker_role": "normal",
                 "text": "Anh ơi, xe em bị sáng đèn engine check, anh xem được không?"},
                {"turn_id": "synth_l09_t02", "speaker_role": "normal",
                 "text": "Được em, em chạy qua garage anh trên đường Lý Thường Kiệt nhé."},
                {"turn_id": "synth_l09_t03", "speaker_role": "normal",
                 "text": "Chi phí kiểm tra khoảng bao nhiêu anh?"},
                {"turn_id": "synth_l09_t04", "speaker_role": "normal",
                 "text": "Kiểm tra OBD miễn phí em, sửa thì tùy lỗi. Em qua chiều nay đi."},
            ],
        },
        # ── 10. Airport pickup ──
        {
            "conversation_id": "synth_l10",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l10_t01", "speaker_role": "normal",
                 "text": "Ba ơi, mai con bay về đến Tân Sơn Nhất lúc 2 giờ chiều."},
                {"turn_id": "synth_l10_t02", "speaker_role": "normal",
                 "text": "OK con, ba ra đón. Con bay hãng gì?"},
                {"turn_id": "synth_l10_t03", "speaker_role": "normal",
                 "text": "Vietnam Airlines ba ạ, cổng quốc nội."},
                {"turn_id": "synth_l10_t04", "speaker_role": "normal",
                 "text": "Ba biết rồi, ba đỗ ở tầng hầm B1. Con ra tới gọi ba."},
            ],
        },
        # ── 11. Study group ──
        {
            "conversation_id": "synth_l11",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l11_t01", "speaker_role": "normal",
                 "text": "Ê, mai thi Giải tích rồi, ôn chung không?"},
                {"turn_id": "synth_l11_t02", "speaker_role": "normal",
                 "text": "Oke, 8 giờ tối ở thư viện nhé."},
                {"turn_id": "synth_l11_t03", "speaker_role": "normal",
                 "text": "Mày làm được mấy bài chương tích phân chưa?"},
                {"turn_id": "synth_l11_t04", "speaker_role": "normal",
                 "text": "Làm được 2/5 thôi, khó quá. Tối giải chung nha."},
            ],
        },
        # ── 12. Wedding planning ──
        {
            "conversation_id": "synth_l12",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l12_t01", "speaker_role": "normal",
                 "text": "Chị ơi, em cưới tháng sau, chị giúp em chọn váy cưới được không?"},
                {"turn_id": "synth_l12_t02", "speaker_role": "normal",
                 "text": "Được chứ em! Chúc mừng em nha. Em thích style nào?"},
                {"turn_id": "synth_l12_t03", "speaker_role": "normal",
                 "text": "Em thích kiểu tối giản, vải satin. Budget khoảng 15 triệu."},
                {"turn_id": "synth_l12_t04", "speaker_role": "normal",
                 "text": "Chị biết tiệm trên Hai Bà Trưng đẹp lắm, cuối tuần mình đi xem nhé."},
            ],
        },
        # ── 13. Pet vet appointment ──
        {
            "conversation_id": "synth_l13",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l13_t01", "speaker_role": "normal",
                 "text": "Bác sĩ ơi, chó em bị nôn từ sáng, em đưa đi khám được không?"},
                {"turn_id": "synth_l13_t02", "speaker_role": "normal",
                 "text": "Được em, em đưa qua phòng khám bất cứ lúc nào. Chó ăn gì hôm qua không?"},
                {"turn_id": "synth_l13_t03", "speaker_role": "normal",
                 "text": "Em cho ăn xương gà, có thể bị hóc."},
                {"turn_id": "synth_l13_t04", "speaker_role": "normal",
                 "text": "Xương gà dễ bị mảnh nhọn. Em đưa qua ngay, bác xem cho nhé."},
            ],
        },
        # ── 14. Office meeting scheduling ──
        {
            "conversation_id": "synth_l14",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l14_t01", "speaker_role": "normal",
                 "text": "Anh Đức ơi, cuộc họp review sprint dời sang 3 giờ chiều nhé."},
                {"turn_id": "synth_l14_t02", "speaker_role": "normal",
                 "text": "OK em, phòng họp nào?"},
                {"turn_id": "synth_l14_t03", "speaker_role": "normal",
                 "text": "Phòng Meeting 2 tầng 5 ạ. Em đã gửi lại invite trên Google Calendar."},
                {"turn_id": "synth_l14_t04", "speaker_role": "normal",
                 "text": "Nhận rồi. Anh chuẩn bị slide demo luôn nhé."},
            ],
        },
        # ── 15. Furniture shopping ──
        {
            "conversation_id": "synth_l15",
            "t1_label": "LEGIT",
            "messages": [
                {"turn_id": "synth_l15_t01", "speaker_role": "normal",
                 "text": "Anh ơi, bộ sofa đăng trên web còn hàng không?"},
                {"turn_id": "synth_l15_t02", "speaker_role": "normal",
                 "text": "Còn chị ạ. Bộ sofa chữ L màu xám, giá 12 triệu."},
                {"turn_id": "synth_l15_t03", "speaker_role": "normal",
                 "text": "Tôi muốn đến showroom xem thực tế."},
                {"turn_id": "synth_l15_t04", "speaker_role": "normal",
                 "text": "Dạ chị qua số 45 Nguyễn Thị Minh Khai, Q.1, mở cửa 8-20 giờ mỗi ngày ạ."},
            ],
        },
    ]


def main():
    """Generate và lưu synthetic conversations."""
    conversations = []
    conversations.extend(build_scam_conversations())
    conversations.extend(build_ambiguous_conversations())
    conversations.extend(build_legit_conversations())

    # Thống kê
    labels = [c["t1_label"] for c in conversations]
    print(f"Generated {len(conversations)} synthetic conversations:")
    print(f"  SCAM:      {labels.count('SCAM')}")
    print(f"  AMBIGUOUS: {labels.count('AMBIGUOUS')}")
    print(f"  LEGIT:     {labels.count('LEGIT')}")

    # Lưu file
    out_dir = os.path.join(STREAMING_ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_conversations.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {out_path}")
    return conversations


if __name__ == "__main__":
    main()
